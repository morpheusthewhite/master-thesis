import graph_tool.all as gt
import treelib
import numpy as np
from typing import Optional, Set
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from scipy.special import softmax

from polarmine.comment import Comment
from polarmine.thread import Thread

KEY_SCORE = "score"
SENTIMENT_MAX_TEXT_LENGTH = 128
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"


class PolarizationGraph:

    """A graph class providing methods for polarization analysis """

    def __init__(self, discussion_trees: list[treelib.Tree]):
        self.graph = gt.Graph()

        # definition of graph property maps
        # edge weights (calculated with sentiment analysis classifier)
        self.weights = self.graph.new_edge_property("double")
        self.times = self.graph.new_edge_property("double")
        self.threads = self.graph.new_edge_property("object")

        # make properties internal
        self.graph.edge_properties["weights"] = self.weights
        self.graph.edge_properties["times"] = self.times
        self.graph.edge_properties["threads"] = self.threads

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
        thread: Thread,
    ):
        edge = self.graph.add_edge(vertex_source, vertex_target)
        sentiment_score = self.sentiment_weight(comment.text)

        self.weights[edge] = sentiment_score
        self.times[edge] = comment.time
        self.threads[edge] = thread

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

    def negative_edges_fraction_thread_dict(self):
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

    def negative_edges_fraction_content_dict(self):
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

    def n_interactions_dict(self):
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

    def edge_sum_n_interactions_dict(self):
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

    def score_from_vertices_index(
        self,
        vertices_index: list[int],
        alpha: int,
        controversial_contents: set = None,
    ) -> float:
        # if not provided find controversial content
        if controversial_contents is None:
            controversial_contents = self.controversial_contents()

        thread_edges_dict = {}
        for vertex in vertices_index:

            for edge in self.graph.vertex(vertex).all_edges():
                edge_weight = self.weights[edge]
                edge_thread = self.threads[edge]
                edge_content = edge_thread.content
                edge_thread_id = edge_thread.url

                if edge_content in controversial_contents:
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

        score = 0

        for thread, n_edges_tuple in thread_edges_dict.items():
            n_negative_edges, n_edges = n_edges_tuple

            if n_negative_edges / n_edges <= alpha:
                # non controversial threads
                score += n_edges

        return score

    def controversial_contents(self, alpha: float) -> set:
        content_dict = self.negative_edges_fraction_content_dict()
        controversial_contents = set()
        for content, fraction in content_dict.items():
            if fraction > alpha:
                controversial_contents.add(content)

        return controversial_contents

    def score_components(self, alpha: int) -> (int, int):
        comp, _ = gt.label_components(self.graph, directed=False)
        controversial_contents = self.controversial_contents(alpha)

        n_components = np.max(comp.a) + 1
        max_score = 0
        max_n_vertices = 0

        for i in range(n_components):
            vertices_index = np.where(comp.a == i)[0]

            score = self.score_from_vertices_index(
                vertices_index, alpha, controversial_contents
            )

            if score > max_score:
                max_score = score
                max_n_vertices = vertices_index.shape[0]

        return max_score, max_n_vertices

    def __find_best_neighbour__(
        self,
        vertices: list[int],
        neighbours: list[int],
        alpha: float,
        controversial_contents: set(),
    ) -> (int, int):
        # keep neighbour increasing more the score
        neighbour_best = -1
        score_neighbour_best = -1

        for neighbour in neighbours:
            score_neighbour = self.score_from_vertices_index(
                vertices + [neighbour],
                alpha,
                controversial_contents,
            )

            if score_neighbour > score_neighbour_best:
                score_neighbour_best = score_neighbour
                neighbour_best = neighbour

        return neighbour_best, score_neighbour_best

    def __find_worst_vertex__(
        self,
        vertices: list[int],
        alpha: float,
        controversial_contents: set(),
    ) -> (int, int):
        # keep neighbour increasing more the score
        vertex_worst = -1
        score_vertex_worst = -1

        for i, vertex in enumerate(vertices):
            # vertices, excluding the current one
            vertices_current = vertices[:i] + vertices[i + 1 :]

            score_vertex = self.score_from_vertices_index(
                vertices_current,
                alpha,
                controversial_contents,
            )

            if score_vertex > score_vertex_worst:
                score_vertex_worst = score_vertex
                vertex_worst = vertex

        return vertex_worst, score_vertex_worst

    def score_greedy(self, alpha: float, beta: float = 1, n_starts: int = -1):
        """Calculate the echo chamber score using the greedy approach

        Args:
            alpha (float): maximum fraction of edges of non controversial content
            beta (float): probability of adding a node along the iterations
            n_starts (int): number of times the algorithm is executed. If -1
            then it is sqrt(num_vertices)
        """
        vertices_index = self.graph.get_vertices()

        if n_starts == -1:
            n_starts = int(np.sqrt(vertices_index.shape[0]))

        controversial_contents = self.controversial_contents(alpha)
        if len(controversial_contents) == 0:
            return 0, 0

        # best score and num_vertices along iterations
        score = -1
        num_vertices = -1

        for i in range(n_starts):
            # sample a node, uniformly
            initial_vertex = np.random.randint(0, vertices_index.shape[0])

            # current set of selected users
            vertices = [initial_vertex]
            # current set of neighbours of the selected users
            neighbours = set(self.graph.get_all_neighbors(initial_vertex))

            while True:
                score_current = self.score_from_vertices_index(
                    vertices, alpha, controversial_contents
                )

                # sample from a bernoulli to decide to add or not
                add_node = np.random.binomial(1, beta)

                if add_node or len(vertices) == 1:
                    (
                        neighbour_best,
                        score_neighbour_best,
                    ) = self.__find_best_neighbour__(
                        vertices, neighbours, alpha, controversial_contents
                    )

                    # if no neighbour increases the score then stop
                    if score_neighbour_best <= score_current:
                        break
                    else:
                        vertices.append(neighbour_best)
                        neighbours.remove(neighbour_best)

                        # add to the list of neighbours the ones of the new node `neighbour`
                        self.__neighbours_merge__(
                            neighbours, neighbour_best, vertices
                        )
                else:
                    # remove the node contributing less to the score
                    (
                        vertex_worst,
                        score_vertex_worst,
                    ) = self.__find_worst_vertex__(
                        vertices, alpha, controversial_contents
                    )

                    vertices.remove(vertex_worst)
                    # remove neighbours of the excluded node
                    self.__neighbours_subtract__(
                        neighbours, vertex_worst, vertices
                    )

            if score_current > score:
                score = score_current
                num_vertices = len(vertices)

        # TODO: add ignore list
        return score, num_vertices

    def __neighbours_merge__(
        self, neighbours: set, vertex: int, vertices: set
    ):
        new_neighbours = self.graph.get_all_neighbours(vertex)
        for new_neighbour in new_neighbours:
            if (
                not new_neighbour in neighbours
                and not new_neighbour in vertices
            ):
                neighbours.add(new_neighbour)

    def __neighbours_subtract__(
        self, neighbours: list[int], vertex_removed: int, vertices: set
    ):
        """remove from neighbours vertices which are no more reachable from vertices

        Args:
            neighbours (set(int)): a list of nodes
            vertex_removed (int): the vertex that has been removed
            vertices (list[int]): vertices of which neighbours should be kept
        """
        neighbours_removed = self.graph.get_all_neighbours(vertex_removed)
        for neighbour_removed in neighbours_removed:

            # if this vertex is listed among `neighbours`, check if it is
            # reachable from another node
            if neighbour_removed in neighbours:

                reachable = False
                for vertex_i in self.graph.get_all_neighbours(
                    neighbour_removed
                ):
                    if vertex_i in vertices:
                        # it is reachable from another node
                        reachable = True
                        break

                if not reachable:
                    neighbours.remove(neighbour_removed)
        return

    @classmethod
    def from_file(cls, filename: str):
        """Creates a PolarizationGraph object from the graph stored in a file

        Args:
            filename (str): filename of the file where the graph is stored
        """
        graph = cls([])
        graph.load(filename)

        return graph
