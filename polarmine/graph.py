from typing import Optional, Set, Iterable
import time
import graph_tool.all as gt
import treelib
import numpy as np
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from scipy.special import softmax

from polarmine.comment import Comment
from polarmine.thread import Thread

SENTIMENT_MAX_TEXT_LENGTH = 128
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
VERTEX_SIZE_SHOW = 50


class InteractionGraph:

    """A graph class providing methods for polarization analysis """

    def __init__(self, discussion_trees: list[treelib.Tree]):
        self.graph = gt.Graph()

        # definition of graph property maps
        # edge weights (calculated with sentiment analysis classifier)
        self.weights = self.graph.new_edge_property("double")
        self.times = self.graph.new_edge_property("int")
        self.threads = self.graph.new_edge_property("object")
        self.comments = self.graph.new_edge_property("string")

        # make properties internal
        self.graph.edge_properties["weights"] = self.weights
        self.graph.edge_properties["times"] = self.times
        self.graph.edge_properties["threads"] = self.threads
        self.graph.edge_properties["comments"] = self.comments

        # initialization of sentiment analysis classifier
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(
            MODEL, normalization=True
        )
        self.sentiment_model = (
            AutoModelForSequenceClassification.from_pretrained(MODEL)
        )

        # dictionary storing user:vertex_index
        self.users = {}

        for discussion_tree in discussion_trees:
            root_id = discussion_tree.root
            root = discussion_tree.nodes[root_id]

            # get the thread, associated to the root node
            thread = root.data

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
                children = discussion_tree.children(node_identifier)

                for child in children:
                    comment = child.data
                    comment_author = child.tag

                    # find the node if it is in the graph
                    comment_vertex = self.get_user_vertex(comment_author)

                    # and add the edge
                    self.add_edge(comment_vertex, node_vertex, comment, thread)

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

    def __kcore_mask__(self, k: int) -> gt.VertexPropertyMap:
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
        thread: Thread,
    ):
        edge = self.graph.add_edge(vertex_source, vertex_target)
        sentiment_score = self.sentiment_weight(comment.text)

        self.weights[edge] = sentiment_score
        self.times[edge] = comment.time
        self.threads[edge] = thread
        self.comments[edge] = comment.text

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
        self.threads = self.graph.edge_properties["threads"]
        self.comments = self.graph.edge_properties["comments"]

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
        communities: Optional[list[int]] = None,
        show_vertices: list[int] = None,
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

        if show_vertices is not None:
            # show passed vertices with a different color
            vertex_fill_color = self.graph.new_vertex_property("string")
            show_vertices = set(show_vertices)
            for vertex in self.graph.vertices():
                if vertex in show_vertices:
                    vertex_fill_color[vertex] = "blue"
                else:
                    vertex_fill_color[vertex] = "gray"
        else:
            # show black vertices otherwise
            vertex_fill_color = "black"

        # use weight to influence layout
        # shift weights to [0, 2] to improve visualization
        # (negative nodes otherwise end up too far apart
        #  weights_positive = self.graph.new_edge_property("int")
        #  weights_positive.a = self.weights.a + 1
        #  pos = gt.sfdp_layout(
        #      self.graph, eweight=weights_positive, p=1.5, C=0.1
        #  )

        if communities is not None:
            node_group_property_map = self.graph.new_vertex_property("int")
            node_group_property_map.a = np.array(communities)

            pos = gt.sfdp_layout(
                self.graph,
                groups=node_group_property_map,
                mu=1000,
                # eweight=weights_positive,
            )
        else:
            pos = None

        gt.graph_draw(
            self.graph,
            pos=pos,
            edge_color=color_property_map,
            edge_pen_width=width_property_map,
            vertex_size=0,
            vertex_fill_color=vertex_fill_color,
            output=output,
        )

    def num_vertices(self):
        return self.graph.num_vertices()

    def num_edges(self):
        return self.graph.num_edges()

    def num_components(self):
        components, _ = gt.label_components(self.graph, directed=False)
        return np.max(components.a) + 1

    def __vertices_components__(self, vertices_index: list[int]) -> np.array:
        components, _ = gt.label_components(self.graph, directed=False)
        return components.a[vertices_index]

    def largest_component_vertices(
        self, vertices_index: list[int]
    ) -> list[int]:
        """Finds the largest components of the given vertices in the graph

        Args:
            vertices_index (list[int]): the list of the vertices index

        Returns:
            list[int]: the list of the vertices index in the largest component
        """
        if len(vertices_index) == 0:
            return []

        vertices_component = self.__vertices_components__(vertices_index)
        n_components = np.max(vertices_component) + 1

        component_max = np.empty((1))

        for i in range(n_components):
            component = np.where(vertices_component == i)[0]

            if component.shape[0] > component_max.shape[0]:
                component_max = component

        return np.array(vertices_index)[component_max]

    def get_echo_chamber_discussion(
        self, vertices_index: Set[int]
    ) -> list[str]:
        """Finds comments posted by users

        Args:
            vertices_index (Set[int]): the set of vertices associated to the users
        """
        comments = {}

        for vertex_index in vertices_index:

            vertex = self.graph.vertex(vertex_index)

            for edge in vertex.out_edges():
                edge_content = self.threads[edge].content

                content_discussion = comments.get(edge_content, [])
                content_discussion.append(
                    [
                        vertex_index,
                        int(edge.target()),
                        self.weights[edge],
                        self.comments[edge],
                    ]
                )

                comments[edge_content] = content_discussion

        return comments

    def components(self) -> list[list[int]]:
        """components.

        Returns:
            list[list[int]]: of list of list, each with the vertex in the connected components
        """
        components, _ = gt.label_components(self.graph, directed=False)
        n_components = np.max(components.a) + 1

        components_list = []
        for i in range(n_components):
            component_indices = np.where(components.a == i)[0]
            components_list.append(component_indices)

        return components_list

    def num_components_from_vertices(self, vertices: list[int]):
        """Count the number of different components associated to the given
        vertices

        Args:
            vertices (list[int]): the list of the indices of the vertices
        """
        components, _ = gt.label_components(self.graph, directed=False)
        return len(set(components.a[vertices]))

    def num_contents(self, alpha: float = -1):
        if alpha != -1:
            contents = self.controversial_contents(alpha)
        else:
            contents = set(map(lambda thread: thread.content, self.threads))
        return len(contents)

    def num_threads(self):
        threads = set(map(lambda thread: thread.url, self.threads))
        return len(threads)

    def negative_edges_fraction(self):

        # verify that a filter exists before cycling
        # since the computation in this case is trivial
        # this apparently does not work
        #  edge_filter_property_map, _ = self.graph.get_edge_filter()
        #  if edge_filter_property_map is None:
        #      return np.sum(self.weights.a < 0) / self.weights.a.shape[0]

        # array containing filtered edges
        edges_weight = np.empty((0,))

        # iterate over index of vertices
        for vertex_index in self.graph.get_vertices():

            # get edges index of the current vertex
            edges_index = self.graph.get_all_edges(
                vertex_index, eprops=[self.weights]
            )

            edges_weight = np.concatenate((edges_weight, edges_index[:, 2]))

        # handle the case in which there are no edges (this may happen if there
        # are only self loops which may be removed
        if edges_weight.shape[0] == 0:
            return 0
        else:
            return np.sum(edges_weight < 0) / edges_weight.shape[0]

    def negative_edges_fraction_thread_dict(self) -> dict:
        """compute the fraction of negative edges for threads which have at
        least one edge

        Returns:
            dict: a dictionary whose key is the thread and value is the
            fraction of negative edges associated to that thread
        """
        # quite inefficient as the cycle is executed in Python
        # this should probably be optimized
        thread_edges_dict = {}

        for edge in self.graph.edges():
            # retrieve the thread and the weight associated with the edge
            edge_thread = self.threads[edge].url
            edge_weight = self.weights[edge]

            current_weights = thread_edges_dict.get(edge_thread, [])
            thread_edges_dict[edge_thread] = current_weights + [edge_weight]

        # array containing the fraction of negative edges for each
        # thread
        fraction_dict = {}

        for thread, weights in thread_edges_dict.items():
            # transform the regular list to a numpy array
            weights_np = np.array(weights)

            negative_fraction = np.sum(weights_np < 0) / weights_np.shape[0]
            fraction_dict[thread] = negative_fraction

        return fraction_dict

    def negative_edges_fraction_content_dict(self) -> dict:
        """compute the fraction of negative edges for contents which have at
        least one edge

        Returns:
            dict: a dictionary whose key is the content and value is the
            fraction of negative edges associated to that content
        """
        # quite inefficient as the cycle is executed in Python
        # this should probably be optimized
        content_edges_dict = {}

        for edge in self.graph.edges():
            # retrieve the content and the weight associated with the edge
            edge_content = self.threads[edge].content
            edge_weight = self.weights[edge]

            current_weights = content_edges_dict.get(edge_content, [])
            content_edges_dict[edge_content] = current_weights + [edge_weight]

        # array containing the fraction of negative edges for each
        # content
        fraction_dict = {}

        for content, weights in content_edges_dict.items():
            # transform the regular list to a numpy array
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

                edge_content = self.threads[edge].content
                if edge_content not in user_contents:
                    user_contents.add(edge_content)

            fidelity = len(user_contents)
            fidelities.append(fidelity)

        return fidelities

    def n_interactions_dict(self) -> dict:
        """compute the number of interactions for contents which have at
        least one edge

        Returns:
            dict: a dictionary whose key is the content and value is the
            number of interactions associated to that content
        """
        n_interactions_dict = {}
        for edge in self.graph.edges():
            edge_content = self.threads[edge].content

            content_n_interactions = n_interactions_dict.get(edge_content, 0)
            content_n_interactions += 1
            n_interactions_dict[edge_content] = content_n_interactions

        return n_interactions_dict

    def n_interaction_values(self):
        n_interactions_dict = self.n_interactions_dict()

        return list(n_interactions_dict.values())

    def edge_sum_n_interactions_dict(self) -> dict:
        """compute the number of edge sum and number of interactions for
        contents which have at least one edge

        Returns:
            dict: a dictionary whose key is the content and value is a the
            tuple (total edge sum, number of interactions) associated to that
            content
        """
        edge_sum_n_interactions_dict = {}
        for edge in self.graph.edges():
            edge_content = self.threads[edge].content
            edge_weight = self.weights[edge]

            n_interactions, weights_sum = edge_sum_n_interactions_dict.get(
                edge_content, (0, 0)
            )
            n_interactions += 1
            weights_sum += edge_weight

            edge_sum_n_interactions_dict[edge_content] = (
                n_interactions,
                weights_sum,
            )

        return edge_sum_n_interactions_dict

    def edge_sum_n_interactions_values(self):
        edge_sum_n_interactions_dict = self.edge_sum_n_interactions_dict()

        return list(edge_sum_n_interactions_dict.values())

    def content_thread_neg_fraction(self):
        # dictionary of dictionaries
        # the key being the content, the value being a dictionary whose key are
        # the threads (associated to the content) and the values are the tuple
        # (n negative edges, number of edges)
        content_thread_edges = {}

        for edge in self.graph.edges():
            edge_thread = self.threads[edge].url
            edge_content = self.threads[edge].content
            edge_weight = self.weights[edge]

            # get the dictionary of threads associated to the content
            threads_dict = content_thread_edges.get(edge_content, {})
            # get the edges tuple associated with the thread
            n_negative_edges, n_edges = threads_dict.get(edge_thread, (0, 0))

            n_edges += 1
            if edge_weight < 0:
                n_negative_edges += 1

            threads_dict[edge_thread] = (n_negative_edges, n_edges)
            content_thread_edges[edge_content] = threads_dict

        content_thread_neg_fraction = {}

        for content, threads_dict in content_thread_edges.items():
            content_thread_neg_fraction[content] = {}

            for thread, edges_tuple in threads_dict.items():
                n_negative_edges, n_edges = edges_tuple
                thread_neg_fraction = n_negative_edges / n_edges

                # compute the fraction of negative nodes for a thread
                content_thread_neg_fraction[content][
                    thread
                ] = thread_neg_fraction

        return content_thread_neg_fraction

    def content_std_dev_dict(self):
        content_thread_neg_fraction = self.content_thread_neg_fraction()
        contents_std_dev = {}

        for content, threads_dict in content_thread_neg_fraction.items():
            content_neg_fractions = list(threads_dict.values())

            # ignore contents without 2 threads at least
            if len(content_neg_fractions) > 2:

                content_neg_fractions_np = np.array(content_neg_fractions)
                content_std_dev = np.std(content_neg_fractions_np)

                contents_std_dev[content] = content_std_dev

        return contents_std_dev

    def global_clustering(self):
        return gt.global_clustering(self.graph)

    def average_degree(self, degree="out", unique=False) -> int:
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

        return counts / np.sum(counts), bins

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

    def __vertices_subthreads_dict__(
        self,
        vertices_index: Iterable[int],
        alpha: float,
        controversial_contents: set = None,
    ) -> dict:
        """Find negativity for all the threads in the graph induced by the
        vertices

        Args:
            vertices_index (list[int]): the list of vertices of the subgraphs
            alpha (int): alpha for controversy definition
            controversial_contents (set): the set of controversial_contents

        Returns:
            dict: dictionary with keys thread and value (num neg. edges, num
            edges) for the subgraphs induced by the vertices
        """
        # if not provided find controversial content
        if controversial_contents is None:
            controversial_contents = self.controversial_contents(alpha)

        thread_edges_dict = {}
        vertices_index = set(vertices_index)
        for vertex in vertices_index:

            # consider only out edges. In this way edges will be counted only
            # once
            for edge in self.graph.vertex(vertex).out_edges():
                edge_weight = self.weights[edge]
                edge_thread = self.threads[edge]
                edge_content = edge_thread.content
                edge_thread_id = edge_thread.url

                source, target = tuple(edge)
                source = int(source)
                target = int(target)

                if (
                    edge_content in controversial_contents
                    and target in vertices_index
                ):
                    n_negative_edges, n_edges = thread_edges_dict.get(
                        edge_thread_id, (0, 0)
                    )

                    if edge_weight < 0:
                        n_negative_edges += 1

                    n_edges += 1
                    thread_edges_dict[edge_thread_id] = (
                        n_negative_edges,
                        n_edges,
                    )

        return thread_edges_dict

    def controversial_contents(self, alpha: float) -> set:
        content_dict = self.negative_edges_fraction_content_dict()
        controversial_contents = set()
        for content, fraction in content_dict.items():
            if fraction > alpha:
                controversial_contents.add(content)

        return controversial_contents

    def alpha_median(self):
        negative_edges_fractions = list(
            self.negative_edges_fraction_content_dict().values()
        )

        alpha_median = np.median(np.array(negative_edges_fractions))
        return alpha_median

    def vertices_positiveness(self) -> np.array:
        """calculate fraction of positive edges for each vertex

        Returns:
            np.array: the fraction of positive edges for vertices
        """
        num_vertices = np.max(self.graph.get_vertices()) + 1

        n_positive_edges = np.zeros((num_vertices))
        n_edges = np.zeros((num_vertices))

        for edge in self.graph.get_edges([self.weights]):
            source = int(edge[0])
            target = int(edge[1])
            weight = edge[2]

            n_edges[source] += 1
            n_edges[target] += 1

            if weight >= 0:
                n_positive_edges[source] += 1
                n_positive_edges[target] += 1

        # set to 1 terms which are to 0 to avoid 0-division
        n_edges = n_edges + (n_edges == 0)
        return n_positive_edges / n_edges

    def positiveness_probabilities(self) -> np.array:
        """calculate the distribution of positive edges among vertices

        Returns:
            np.array: the parameters of a categorical distribution, based on
            the fraction of negative edges of the vertex
        """
        vertices_positiveness = self.vertices_positiveness()
        total_positiveness = np.sum(vertices_positiveness)

        return vertices_positiveness / total_positiveness

    def score_densest_nc_subgraph(
        self, alpha: float, simple: bool = True
    ) -> (float, list[int]):
        raise NotImplementedError

    def o2_bff_dcs_am(self, alpha: float, k: int) -> (int, list[int]):
        raise NotImplementedError

    def select_echo_chamber(
        self,
        alpha: float,
        vertices_index: list[int],
        controversial_contents: set = None,
    ):

        edge_filter = self.graph.new_edge_property("bool", val=False)
        vertex_filter = self.graph.new_vertex_property("bool", val=False)

        # TODO: fix me
        raise NotImplementedError
        nc_threads = []
        #  _, nc_threads = score_from_vertices_index(
        #      self, vertices_index, alpha, controversial_contents
        #  )

        # use a set for faster search
        vertices_index_set = set(vertices_index)
        nc_threads = set(nc_threads)

        for vertex_index in vertices_index_set:
            vertex = self.graph.vertex(vertex_index)
            vertex_filter[vertex] = True

            for edge in vertex.out_edges():
                # check if both the thread is non controversial and the target
                # node is in the echo chamber
                if (
                    self.threads[edge].url in nc_threads
                    and int(edge.target()) in vertices_index_set
                ):
                    edge_filter[edge] = True

        self.graph.set_vertex_filter(vertex_filter)
        self.graph.set_edge_filter(edge_filter)

        return

    def __edge_sample__(
        self,
        i: int,
        j: int,
        node_groups: list[int],
        omega_positive: np.array,
        omega_negative: np.array,
        theta: float,
        group_activation: bool,
        active_communities: set = None,
        user_is_active: np.array = None,
    ):
        # get the groups of both vertices
        group_r = node_groups[i]
        group_s = node_groups[j]

        # get probabilities between group r and s
        omega_positive_rs = omega_positive[group_r, group_s]
        omega_negative_rs = omega_negative[group_r, group_s]

        if group_activation:
            if (
                group_r not in active_communities
                or group_s not in active_communities
            ):
                omega_positive_rs = omega_negative_rs * theta
                omega_negative_rs = omega_negative_rs * theta
        else:
            # if one of the 2 nodes are not active reduce the
            # probabilities
            if not user_is_active[i] or not user_is_active[j]:
                omega_positive_rs = omega_negative_rs * theta
                omega_negative_rs = omega_negative_rs * theta

        omega_null_rs = 1 - omega_positive_rs - omega_negative_rs

        # draw from the multinomial with the given parameters
        edge_outcome = np.where(
            np.random.multinomial(
                n=1,
                pvals=[
                    omega_positive_rs,
                    omega_negative_rs,
                    omega_null_rs,
                ],
            )
        )[0][0]

        return edge_outcome

    def __thread_sample__(
        self,
        thread_id: str,
        content_id: str,
        nodes: list[gt.Vertex],
        node_groups: list[int],
        omega_positive: np.array,
        omega_negative: np.array,
        theta: float,
        group_activation: bool,
        active_communities: set = None,
        node_is_active: list[int] = None,
    ):
        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                if i < j:
                    edge_outcome = self.__edge_sample__(
                        i,
                        j,
                        node_groups,
                        omega_positive,
                        omega_negative,
                        theta,
                        group_activation,
                        active_communities,
                        node_is_active,
                    )

                    if edge_outcome != 2:
                        # an edge must be added
                        edge = self.graph.add_edge(node_i, node_j)
                        self.threads[edge] = Thread(
                            thread_id, None, None, None, content_id
                        )

                        if edge_outcome == 0:
                            self.weights[edge] = +1
                        else:
                            self.weights[edge] = -1

    def generate1(
        self,
        n_nodes: np.array,
        n_threads: np.array,
        omega_positive: np.array,
        omega_negative: np.array,
        theta: float,
        n_active_communities: int,
    ):
        n_communities = len(n_nodes)

        node_groups = []
        for i, n_group_nodes in enumerate(n_nodes):
            node_groups += [i] * n_group_nodes

        # add the needed number of vertices to the graph
        nodes = [vertex for vertex in self.graph.add_vertex(n=np.sum(n_nodes))]

        for k in range(n_threads):
            active_communities = set(
                np.random.choice(n_communities, 2, replace=False)
            )
            self.__thread_sample__(
                str(k),
                str(k),
                nodes,
                node_groups,
                omega_positive,
                omega_negative,
                theta,
                True,
                active_communities,
            )

    def generate2(
        self,
        n_nodes: np.array,
        n_threads: int,
        phi: np.array,
        omega_positive: np.array,
        omega_negative: np.array,
        theta: float,
        beta_a: float,
        beta_n: float,
    ):

        node_groups = []
        for i, n_group_nodes in enumerate(n_nodes):
            node_groups += [i] * n_group_nodes

        follow_graph = gt.generate_sbm(node_groups, phi)

        # add the needed number of vertices to the graph
        nodes = [vertex for vertex in self.graph.add_vertex(n=np.sum(n_nodes))]

        for k in range(n_threads):

            # sample if a node is active or not, i.e. from a categorical with
            # probability (beta_a, 1-beta_a)
            node_is_active = np.random.choice(
                2, len(node_groups), p=[1 - beta_a, beta_a]
            )

            active_queue = list(np.where(node_is_active)[0])

            # propagate activation through neighbours
            while len(active_queue) > 0:
                active_node = active_queue.pop(0)

                node_neighbours = follow_graph.get_all_neighbors(active_node)

                # sample once for all the neighbours of the node, even if some
                # of them may already be active
                get_activated = np.random.choice(
                    2, node_neighbours.shape[0], p=[1 - beta_n, beta_n]
                )

                for j, neighbour in enumerate(node_neighbours):
                    if not node_is_active[neighbour] and get_activated[j]:
                        node_is_active[neighbour] = 1
                        active_queue.append(neighbour)

            self.__thread_sample__(
                str(k),
                str(k),
                nodes,
                node_groups,
                omega_positive,
                omega_negative,
                theta,
                False,
                None,
                node_is_active,
            )

    def get_positive_edges(self):
        edges = self.graph.get_edges([self.weights])
        positive_edges = edges[edges[:, 2] >= 0]
        return positive_edges

    def is_induced_edge(self, vertices: set, threads: set):
        is_induced_property = self.graph.new_edge_property("bool")

        for edge in self.graph.edges():
            source, target = tuple(edge)
            thread = self.threads[edge].url

            if (
                source in vertices
                and target in vertices
                and (len(threads) == 0 or thread in threads)
            ):
                is_induced_property[edge] = True

        return is_induced_property

    def threads_time_span(self):
        start_time = int(np.min(self.times.a))
        end_time = int(np.max(self.times.a))

        start_time_str = time.ctime(start_time)
        end_time_str = time.ctime(end_time)
        return start_time_str, end_time_str

    def clear_filters(self):
        self.graph.clear_filters()
        return

    def shuffle(self):
        pass

    @classmethod
    def from_model1(
        cls,
        n_nodes: np.array,
        n_threads: int,
        omega_positive: np.array,
        omega_negative: np.array,
        theta: float,
        n_active_communities: int,
    ):
        """Creates a InteractionGraph object from the given model parameters

        Args:
            n_nodes (np.array): a numpy array with the number of elements in each class
            omega_positive (np.array): a numpy 2D array where element ij
            contains the probability of positive edge between class i and j
            omega_positive (np.array): a numpy 2D array where element ij
            contains the probability of negative edge between class i and j
            theta: parameter controlling the probability of interacting between
            inactive communities
            n_active_communities: number of active communities in each content
        """
        graph = cls([])
        graph.generate1(
            n_nodes,
            n_threads,
            omega_positive,
            omega_negative,
            theta,
            n_active_communities,
        )

        return graph

    @classmethod
    def from_model2(
        cls,
        n_nodes: np.array,
        n_threads: int,
        phi: np.array,
        omega_positive: np.array,
        omega_negative: np.array,
        theta: float,
        beta_a: float,
        beta_n: float,
    ):
        """Creates a InteractionGraph object from the given model parameters

        Args:
            n_nodes (np.array): a numpy array with the number of elements in each class
            omega_positive (np.array): a numpy 2D array where element ij
            contains the probability of positive edge between class i and j
            omega_positive (np.array): a numpy 2D array where element ij
            contains the probability of negative edge between class i and j
            theta: parameter controlling the probability of interacting between
            inactive communities
            n_active_communities: number of active communities in each content
        """
        graph = cls([])
        graph.generate2(
            n_nodes,
            n_threads,
            phi,
            omega_positive,
            omega_negative,
            theta,
            beta_a,
            beta_n,
        )

        return graph

    @classmethod
    def from_file(cls, filename: str):
        """Creates a InteractionGraph object from the graph stored in a file

        Args:
            filename (str): filename of the file where the graph is stored
        """
        graph = cls([])
        graph.load(filename)

        return graph
