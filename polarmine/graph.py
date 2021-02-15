import graph_tool.all as gt
import treelib
import numpy as np
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from scipy.special import softmax

from polarmine.comment import Comment
from polarmine.content import Content

KEY_SCORE = "score"
SENTIMENT_MAX_TEXT_LENGTH = 128
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"


class PolarizationGraph:

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
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(
            MODEL, normalization=True
        )
        self.sentiment_model = (
            AutoModelForSequenceClassification.from_pretrained(MODEL)
        )

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
                    comment_author = child.tag

                    # find the node if it is in the graph
                    comment_vertex = self.get_user_vertex(comment_author)

                    # and add the edge
                    self.add_edge(
                        comment_vertex, node_vertex, comment, content
                    )

                    # equeue this child
                    queue.append(child)

        # precompute kcore-decomposition
        self.kcore = gt.kcore_decomposition(self.graph)

    def __kcore_mask__(self, k: int) -> gt.EdgePropertyMap:
        """mask nodes which are not in k-core

        Args:
            k (int): k

        Returns:
            gt.EdgePropertyMap: a boolean property in which nodes out
            of the k-core are masked out
        """
        mask = self.graph.new_vertex_property("bool")
        mask.a = self.kcore.a >= k

        return mask

    def select_kcore(self, k) -> None:
        """select kcore of the graph. Function called after a call to this
        function will operate only on its kcore

        Args:
            k
        """
        kmask = self.__kcore_mask__(k)
        self.graph.set_vertex_filter(kmask)

    def add_edge(
        self,
        vertex_source: gt.Vertex,
        vertex_target: gt.Vertex,
        comment: Comment,
        content: Content,
    ):
        edge = self.graph.add_edge(vertex_source, vertex_target)
        sentiment_score = self.sentiment_weight(comment.text)

        self.weights[edge] = sentiment_score
        self.times[edge] = comment.time
        self.contents[edge] = content

    def sentiment_weight(self, text):

        # return a list of dictionary of this type
        # array([-0.8606459 ,  0.6321694 ,  0.24943551], dtype=float32)
        # scores[0] is the 'NEGATIVE' label, scores[2] is the 'POSITIVE'
        try:
            tokens = self.sentiment_tokenizer(text, return_tensors="pt")
            scores = self.sentiment_model(**tokens)[0][0].detach().numpy()
        except IndexError:
            # text too long
            tokens = self.sentiment_tokenizer(
                text[:SENTIMENT_MAX_TEXT_LENGTH], return_tensors="pt"
            )
            scores = self.sentiment_model(**tokens)[0][0].detach().numpy()

        # the score returned by the classifier is the highest between the 2
        # probabilities
        # to [0, 1] and using the label as sign
        probabilities = softmax(scores)
        probability_positive = probabilities[1] + probabilities[2]

        if probabilities[0] < probability_positive:
            return probability_positive
        else:
            return -probabilities[0]

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

    def load(self, filename: str) -> None:
        """load the graph from a file

        Args:
            filename (str): filename of the file where the graph is stored
        """
        if not filename.endswith(".gt"):
            filename = filename + ".gt"

        self.graph = gt.load_graph(filename)

        # load class attributes. Note: self.users is not initialized as it
        # not considered important
        self.weights = self.graph.edge_properties["weights"]
        self.times = self.graph.edge_properties["times"]
        self.contents = self.graph.edge_properties["contents"]

        # precompute kcore-decomposition
        self.kcore = gt.kcore_decomposition(self.graph)

    def dump(self, filename: str) -> None:
        """dump the current graph

        Args:
            filename (str): filename of the file where the graph will be stored
        """
        if not filename.endswith(".gt"):
            filename = filename + ".gt"

        self.graph.save(filename)

    def draw(
        self,
        edge_color: bool = True,
        edge_width: bool = True,
    ) -> None:
        """draw the graph

        Args:
            edge_color (bool): if True color edges according to weight sign
            edge_width (bool): if True draw edges according to weight abs value
        """
        if edge_color:
            color_property_map = self.graph.new_edge_property("string")

            for edge in self.graph.edges():
                color_property_map[edge] = (
                    "red" if self.weights[edge] < 0 else "green"
                )
        else:
            color_property_map = None

        if edge_width:
            width_property_map = self.graph.new_edge_property("double")
            width_property_map.a = np.abs(self.weights.a)
        else:
            width_property_map = None

        gt.graph_draw(
            self.graph,
            edge_color=color_property_map,
            edge_pen_width=width_property_map,
            vertex_size=2,
        )

    def summarize(self):
        print(
            f"The graph has {self.graph.num_vertices()} vertices and {self.graph.num_edges()} edges"
        )

    def negative_edges_fraction(self):
        num_edges = self.graph.num_edges()
        edge_filter_property_map, _ = self.graph.get_edge_filter()

        # verify that a filter exists before using the property
        if edge_filter_property_map is not None:
            num_negative_edges = np.sum(
                (edge_filter_property_map.a * self.weights.a) < 0
            )
        else:
            num_negative_edges = np.sum(self.weights.a < 0)

        return num_negative_edges / num_edges

    def global_clustering(self):
        return gt.global_clustering(self.graph)

    def degree_histogram_total(self):
        return gt.vertex_hist(
            self.graph, self.graph.degree_property_map("total")
        )

    def kcore_size(self):
        num_vertices = self.graph.num_vertices(True)
        num_vertices_kcore = self.graph.num_vertices()
        return num_vertices_kcore / num_vertices

    @classmethod
    def from_file(cls, filename: str):
        """Creates a PolarizationGraph object from the graph stored in a file

        Args:
            filename (str): filename of the file where the graph is stored
        """
        graph = cls([])
        graph.load(filename)

        return graph
