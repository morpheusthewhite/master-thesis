import graph_tool.all as gt
import treelib
from transformers import pipeline

from polarmine.comment import Comment
from polarmine.content import Content

LABEL_NEGATIVE="NEGATIVE"
LABEL_POSITIVE="POSITIVE"
KEY_SCORE="score"
KEY_LABEL="label"
SENTIMENT_MAX_TEXT_LENGTH=2048


class PolarizationGraph():

    """A graph class providing methods for polarization analysis """

    def __init__(self, threads: list[treelib.Tree]):
        self.graph = gt.Graph()

        # definition of graph property maps
        # edge weights (calculated with sentiment analysis classifier)
        self.weights = self.graph.new_edge_property("double")
        self.times = self.graph.new_edge_property("double")
        self.contents = self.graph.new_edge_property("object")

        # make properties internal
        self.graph.edge_properties["weights"] = self.weights
        self.graph.edge_properties["times"] = self.times
        self.graph.edge_properties["contents"] = self.contents

        # initialization of sentiment analysis classifier
        # the default one is used, but if better alternatives are found
        # they could be used instead
        self.cls_sentiment_analysis = pipeline("sentiment-analysis")

        # dictionary storing user:vertex_index
        self.users = {}

        for thread in threads:
            root_id = thread.root
            root = thread.nodes[root_id]

            # get the content, associated to the root node
            content = root.data

            # TODO: do you want to add the root to the graph? Seems so

            # iterate over all other nodes
            # initially the queue will contain just the root node children
            queue = [root]

            while len(queue) > 0:
                # remove one element from the queue
                node = queue.pop(0)
                node_identifier = node.identifier

                # get/create the corresponding vertex
                node_author = node.tag
                node_vertex = self.get_user_vertex(node_author)

                # children of the current node
                children = thread.children(node_identifier)

                for child in children:
                    comment = child.data
                    comment_author = node.tag

                    # find the node if it is in the graph
                    comment_vertex = self.get_user_vertex(comment_author)

                    # and add the edge
                    self.add_edge(comment_vertex, node_vertex, comment,
                                  content)

                    # equeue this child
                    queue.append(child)

    def add_edge(self, vertex_source: gt.Vertex, vertex_target: gt.Vertex,
                 comment: Comment, content: Content):
        edge = self.graph.add_edge(vertex_source, vertex_target)

        # return a list of dictionary of this type
        # [{'label': 'NEGATIVE', 'score': 0.8729901313781738}]
        try:
            sentiment_dictionary = self.cls_sentiment_analysis(comment.text)[0]
        except IndexError:
            # text too long
            sentiment_dictionary = \
                    self.cls_sentiment_analysis(
                        comment.text[:SENTIMENT_MAX_TEXT_LENGTH])[0]

        # the score returned by the classifier is the highest between the 2
        # probabilities and so it is always >= 0.5
        # it is mapped "continously" in [-1, 1] by expanding [0.5, 1]
        # to [0, 1] and using the label as sign
        sentiment_unsigned_score = (sentiment_dictionary[KEY_SCORE] - 0.5) * 2
        if sentiment_dictionary[KEY_LABEL] == LABEL_NEGATIVE:
            sentiment_score = - sentiment_unsigned_score
        else:
            sentiment_score = sentiment_unsigned_score

        self.weights[edge] = sentiment_score
        self.times[edge] = comment.time
        self.contents[edge] = content

    def get_user_vertex(self, user: str) -> gt.Vertex:
        vertex_index = self.users.get(user)

        if vertex_index is None:
            vertex = self.graph.add_vertex()

            # get the index and add it to the dictionary
            vertex_index = self.graph.vertex_index[vertex]
            self.users[user] = vertex_index
        else:
            # retrieve the vertex object from the graph
            vertex = self.graph.vertex(vertex_index)

        return vertex

    def draw(self):
        gt.graph_draw(self.graph)


