import graph_tool.all as gt
import treelib
import numpy as np
from typing import Optional
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

        self.self_loop_mask = self.graph.new_edge_property("bool")
        self.self_loop_mask.a = (
            1 - gt.label_self_loops(self.graph, mark_only=True).a
        )

        # precompute kcore-decomposition
        self.__kcore__ = self.__kcore_decomposition__()

    def __kcore_decomposition__(self):
        """Wrapper for grapt_tool kcore_decomposition excluding self edges"""

        self.graph.set_edge_filter(self.self_loop_mask)
        kcore = gt.kcore_decomposition(self.graph)
        self.graph.set_edge_filter(None)

        return kcore

    def __kcore_mask__(self, k: int) -> gt.EdgePropertyMap:
        """mask nodes which are not in k-core

        Args:
            k (int): k

        Returns:
            gt.EdgePropertyMap: a boolean property in which nodes out
            of the k-core are masked out
        """
        mask = self.graph.new_vertex_property("bool")
        mask.a = self.__kcore__.a >= k

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

    def get_user_vertex(self, user: int) -> gt.Vertex:
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

        # compute self-loop mask
        self.self_loop_mask = self.graph.new_edge_property("bool")
        self.self_loop_mask.a = (
            1 - gt.label_self_loops(self.graph, mark_only=True).a
        )

        # precompute kcore-decomposition
        self.__kcore__ = self.__kcore_decomposition__()

    def dump(self, filename: str) -> None:
        """dump the current graph

        Args:
            filename (str): filename of the file where the graph will be stored
        """
        if not filename.endswith(".gt"):
            filename = filename + ".gt"

        self.graph.save(filename)

    def remove_self_loops(self):
        gt.remove_self_loops(self.graph)

    def draw(
        self,
        edge_color: bool = True,
        edge_width: bool = True,
        output: Optional[str] = None,
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

        # use weight to influence layout
        # shift weights to [0, 2] to improve visualization
        # (negative nodes otherwise end up too far apart
        #  weights_positive = self.graph.new_edge_property("int")
        #  weights_positive.a = self.weights.a + 1
        #  pos = gt.sfdp_layout(
        #      self.graph, eweight=weights_positive, p=1.5, C=0.1
        #  )

        gt.graph_draw(
            self.graph,
            #  pos=pos,
            edge_color=color_property_map,
            edge_pen_width=width_property_map,
            vertex_size=0,
            vertex_fill_color="black",
            output=output,
        )

    def num_vertices(self):
        return self.graph.num_vertices()

    def num_edges(self):
        return self.graph.num_edges()

    def negative_edges_fraction(self):

        # verify that a filter exists before cycling
        edge_filter_property_map, _ = self.graph.get_edge_filter()
        if edge_filter_property_map is None:
            return np.sum(self.weights.a < 0) / self.weights.a.shape[0]

        # array containing filtered edges
        edges_weight = np.empty((0,))

        # iterate over index of vertices
        for vertex_index in self.graph.get_vertices():

            # get edges index of the current vertex
            edges_index = self.graph.get_all_edges(
                vertex_index, eprops=[self.weights]
            )

            edges_weight = np.concatenate((edges_weight, edges_index[:, 2]))

        return np.sum(edges_weight < 0) / edges_weight.shape[0]

    def negative_edges_fraction_dict(self):
        # quite inefficient as the cycle is executed in Python
        # this should probably be optimized
        content_edges_dict = {}

        for edge in self.graph.edges():
            content = self.contents[edge].url
            weight = self.weights[edge]

            current_weights = content_edges_dict.get(content, [])
            content_edges_dict[content] = current_weights + [weight]

        # array containing the fraction of negative edges for each
        # content
        fraction_dict = {}

        for content, weights in content_edges_dict.items():
            weights_np = np.array(weights)

            negative_fraction = np.sum(weights_np < 0) / weights_np.shape[0]
            fraction_dict[content] = negative_fraction

        return fraction_dict

    def fidelity_values(self):
        fidelities = []

        # iterate over index of vertices
        for vertex_index in self.graph.get_vertices():

            # get edges index of the current vertex
            edges_index = self.graph.get_all_edges(vertex_index)

            user_contents = set()
            for edge_index in edges_index:
                edge = self.graph.edge(edge_index[0], edge_index[1])

                edge_content = self.contents[edge]
                if edge_content not in user_contents:
                    user_contents.add(edge_content)

            fidelity = len(user_contents)
            fidelities.append(fidelity)

        return fidelities

    def global_clustering(self):
        return gt.global_clustering(self.graph)

    def average_degree(self, degree="total", unique=False) -> int:
        """compute average degree. Consider only filtered nodes

        Args:
            degree: which degree to consider. Either "total", "in" or "out"
            unique: if True does not consider multiedges

        Returns:
            int: the average degree
        """
        if not unique:
            return gt.vertex_average(self.graph, degree)[0]
        else:
            degree_accumulator = 0

            # iterate over filtered vertices
            for vertex in self.graph.vertices():

                if degree == "total":
                    neighbors = self.graph.get_all_neighbors(vertex)
                elif degree == "in":
                    neighbors = self.graph.get_in_neighbors(vertex)
                elif degree == "out":
                    neighbors = self.graph.get_out_neighbors(vertex)
                else:
                    raise Exception("Invalid degree parameter value")

                degree_accumulator += np.unique(neighbors).shape[0]

            return degree_accumulator / self.graph.num_vertices()

    def degree_distribution(self, degree="total") -> (list[int], list[int]):
        """compute cumulative degree distribution (takes into account multiedges)

        Args:
            degree: which degree to consider. Either "total", "in" or "out"

        Returns:
            (list[int], list[int]): the cumulative probability of the degree for
            each value and the list of values
        """
        counts, bins = gt.vertex_hist(self.graph, degree)

        # since bins represent the edge of the bins,
        # the last one is removed (in order to make them equal in number to
        # the counts). In this way each bin will be represented by its start
        bins = bins[: bins.shape[0] - 1]

        cum_counts = np.cumsum(counts[::-1])[::-1]
        cum_probabilities = cum_counts / np.sum(counts)

        return cum_probabilities, bins

    def degree_histogram(self, degree="total") -> (list[int], list[int]):
        """compute histogram of the degree

        Args:
            degree: which degree to consider. Either "total", "in" or "out"

        Returns:
            (list[int], list[int]): the list of number of elements in each bin
            and the start of each bin
        """
        counts, bins = gt.vertex_hist(self.graph, degree)

        # since bins represent the edge of the bins,
        # the last one is removed (in order to make them equal in number to
        # the counts). In this way each bin will be represented by its start
        bins = bins[: bins.shape[0] - 1]

        return counts, bins

    def degree_values(self, degree: str = "total") -> np.array:
        """returns degree of vertices

        Args:
            degree: which degree to consider. Either "total", "in" or "out"

        Returns:
            np.array: a numpy array containing the degree of the edges
        """
        return np.array(self.graph.degree_property_map(degree).a)

    def kcore_size(self):
        num_vertices = self.graph.num_vertices(True)
        num_vertices_kcore = self.graph.num_vertices()
        return num_vertices_kcore / num_vertices

    def average_shortest_path_length(self):
        distances_property_map = gt.shortest_distance(self.graph)

        # get max integer that can be represented with an int32
        ii32 = np.iinfo(np.int32)
        maxint32 = ii32.max

        # accumulator over sum of all distances which are not infinite
        distance_accumulator = 0
        # accumulator over number of distances which are not infinite
        n_accumulator = 0

        for vertex in self.graph.vertices():

            # distance of a single vertex from all other nodes
            # vertex_distances in a numpy array containing distances
            # from all nodes
            vertex_distances = np.array(distances_property_map[vertex])

            # consider only reachable node and exclude distance from node itself
            reachable = (
                1 - (vertex_distances == maxint32) - (vertex_distances == 0)
            )

            distance_accumulator += np.sum(reachable * vertex_distances)
            n_accumulator += np.sum(reachable)

        return distance_accumulator / n_accumulator

    def median_shortest_path_length(self):
        distances_property_map = gt.shortest_distance(self.graph)

        # get max integer that can be represented with an int32
        ii32 = np.iinfo(np.int32)
        maxint32 = ii32.max

        # numpy array containing all distances, initially empty
        distances = np.ndarray(0)

        for vertex in self.graph.vertices():

            # distance of a single vertex from all other nodes
            # vertex_distances in a numpy array containing distances
            # from all nodes
            vertex_distances = np.array(distances_property_map[vertex])

            # consider only reachable node
            reachable = (
                1 - (vertex_distances == maxint32) - (vertex_distances == 0)
            )

            # indeces of reachable vertixes from the current one
            reachable_vertices = np.where(reachable)[0]

            valid_distances = vertex_distances[reachable_vertices]
            distances = np.concatenate((distances, valid_distances))

        return np.median(distances)

    @classmethod
    def from_file(cls, filename: str):
        """Creates a PolarizationGraph object from the graph stored in a file

        Args:
            filename (str): filename of the file where the graph is stored
        """
        graph = cls([])
        graph.load(filename)

        return graph
