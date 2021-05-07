import graph_tool.all as gt
import treelib
import numpy as np
import pulp
from typing import Optional
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from scipy.special import softmax
from sklearn import metrics

from polarmine.collectors.twitter_collector import TwitterCollector
from polarmine.comment import Comment
from polarmine.thread import Thread
from polarmine import densest

KEY_SCORE = "score"
SENTIMENT_MAX_TEXT_LENGTH = 128
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"


class PolarizationGraph:

    """A graph class providing methods for polarization analysis """

    def __init__(self, discussion_trees: list[treelib.Tree]):
        self.graph = gt.Graph()

        # definition of graph property maps
        # edge weights (calculated with sentiment analysis classifier)
        self.screen_names = self.graph.new_vertex_property("string")
        self.labels = self.graph.new_vertex_property("int")
        self.weights = self.graph.new_edge_property("double")
        self.times = self.graph.new_edge_property("double")
        self.threads = self.graph.new_edge_property("object")

        # make properties internal
        self.graph.vertex_properties["screen_names"] = self.screen_names
        self.graph.vertex_properties["labels"] = self.labels
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
            self.screen_names[vertex] = user
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
        self.screen_names = self.graph.vertex_properties["screen_names"]
        self.labels = self.graph.vertex_properties["labels"]
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
        communities: Optional[list[int]] = None,
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
            vertex_fill_color="black",
            output=output,
        )

    def num_vertices(self):
        return self.graph.num_vertices()

    def num_edges(self):
        return self.graph.num_edges()

    def num_components(self):
        components, _ = gt.label_components(self.graph)
        return np.max(components.a) + 1

    def num_contents(self, alpha: float = -1):
        if alpha != -1:
            contents = self.controversial_contents(alpha)
        else:
            contents = set(map(lambda thread: thread.content, self.threads))
        return len(contents)

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

    def __vertices_subthreads_dict__(
        self,
        vertices_index: list[int],
        alpha: int,
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

    def score_from_vertices_index(
        self,
        vertices_index: list[int],
        alpha: int,
        controversial_contents: set = None,
    ) -> (float, list[str]):
        thread_edges_dict = self.__vertices_subthreads_dict__(
            vertices_index, alpha, controversial_contents
        )
        score = 0
        nc_threads = []

        for thread, n_edges_tuple in thread_edges_dict.items():
            n_negative_edges, n_edges = n_edges_tuple

            if n_negative_edges / n_edges <= alpha:
                # non controversial threads
                score += n_edges
                nc_threads.append(thread)

        return score, nc_threads

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

    def score_components(self, alpha: int) -> (int, list[int], int):
        comp, _ = gt.label_components(self.graph, directed=False)
        controversial_contents = self.controversial_contents(alpha)

        n_components = int(np.max(comp.a) + 1)
        max_score = 0
        max_users_index = []
        max_n_nc_threads = 0

        for i in range(n_components):
            vertices_index = np.where(comp.a == i)[0]

            score, nc_threads = self.score_from_vertices_index(
                vertices_index, alpha, controversial_contents
            )
            n_nc_threads = len(nc_threads)

            if score > max_score:
                max_score = score
                max_users_index = vertices_index
                max_n_nc_threads = n_nc_threads

        return max_score, max_users_index, max_n_nc_threads

    def __find_best_neighbour__(
        self,
        vertices: list[int],
        neighbours: list[int],
        alpha: float,
        controversial_contents: set(),
    ) -> (int, int):
        # keep neighbours increasing more the score
        neighbours_best = []
        score_neighbour_best = -1

        for neighbour in neighbours:
            (score_neighbour, _,) = self.score_from_vertices_index(
                vertices + [neighbour],
                alpha,
                controversial_contents,
            )

            if score_neighbour > score_neighbour_best:
                score_neighbour_best = score_neighbour
                neighbours_best = [neighbour]
            elif score_neighbour == score_neighbour_best:
                neighbours_best.append(neighbour)

        # sample one node among the many whose addition produce the highest score
        neighbour_best_index = np.random.randint(0, len(neighbours_best))
        neighbour_best = neighbours_best[neighbour_best_index]

        return neighbour_best, score_neighbour_best

    def __find_worst_vertex__(
        self,
        vertices: list[int],
        alpha: float,
        controversial_contents: set(),
    ) -> (int, int):
        # keep neighbour increasing more the score
        vertices_worst = []
        score_vertex_worst = -1

        for i, vertex in enumerate(vertices):
            # vertices, excluding the current one
            vertices_current = vertices[:i] + vertices[i + 1 :]

            score_vertex, _ = self.score_from_vertices_index(
                vertices_current,
                alpha,
                controversial_contents,
            )

            if score_vertex > score_vertex_worst:
                score_vertex_worst = score_vertex
                vertices_worst = [vertex]
            elif score_vertex == score_vertex_worst:
                vertices_worst.append(vertex)

        # sample one node among the many whose removal produce the highest score
        vertex_worst_index = np.random.randint(0, len(vertices_worst))
        vertex_worst = vertices_worst[vertex_worst_index]
        return vertex_worst, score_vertex_worst

    def score_greedy_beta(
        self,
        alpha: float,
        beta: float = 1,
        n_starts: int = -1,
        positiveness_samples: bool = True,
    ) -> (int, list[int]):
        """Calculate the echo chamber score using the beta greedy approach

        Args:
            alpha (float): maximum fraction of edges of non controversial content
            beta (float): probability of adding a node along the iterations
            n_starts (int): number of times the algorithm is executed. If -1
            then it is sqrt(num_vertices)
        """
        vertices_index = self.graph.get_vertices()

        if n_starts == -1:
            n_starts = int(np.sqrt(vertices_index.shape[0]))

        # if there are no controversial contents avoid executing the algorithm
        controversial_contents = self.controversial_contents(alpha)
        if len(controversial_contents) == 0:
            return 0, [], 0

        # best score and corresponding users along iterations
        score = -1
        users_index = []
        nc_threads = 0

        if positiveness_samples:
            sample_probabilities = self.positiveness_probabilities()

        for _ in range(n_starts):
            # list containing nodes which are temporarily ignored
            vertices_ignore = []

            if positiveness_samples:
                initial_vertex_index = np.where(
                    np.random.multinomial(1, sample_probabilities)
                )[0][0]
            else:
                # sample a node, uniformly
                initial_vertex_index = np.random.randint(
                    0, vertices_index.shape[0]
                )

            initial_vertex = vertices_index[initial_vertex_index]

            # current set of selected users
            vertices = [initial_vertex]
            # current set of neighbours of the selected users
            neighbours = set(self.graph.get_all_neighbors(initial_vertex))

            score_current = -1
            # terminate the algorithm if no neighbour can be added
            while len(neighbours) > 0:

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

                        score_current = score_neighbour_best
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

                    if len(vertices_ignore) == 0:
                        vertices_ignore.append(None)
                    vertices_ignore.append(vertex_worst)

                    score_current = score_vertex_worst

                if not len(vertices_ignore) == 0:
                    vertex_ignored = vertices_ignore.pop(0)
                    if vertex_ignored is not None:
                        neighbours.add(vertex_ignored)

            if score_current > score:
                score = score_current
                users_index = vertices

        # calculate number of controversial threads
        score, nc_threads = self.score_from_vertices_index(
            users_index, alpha, controversial_contents
        )
        n_nc_threads = len(nc_threads)
        return score, users_index, n_nc_threads

    def __neighbours_merge__(
        self, neighbours: set, vertex: int, vertices: set
    ):
        new_neighbours = self.graph.get_all_neighbours(vertex)
        for new_neighbour in new_neighbours:
            if new_neighbour not in vertices:
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

    def score_greedy_peeling(self, alpha: float) -> (int, list[int]):
        """Calculate the echo chamber score using the "peeling" greedy approach

        Args:
            alpha (float): maximum fraction of edges of non controversial content
        """
        vertices_index = list(self.graph.get_vertices())

        # if there are no controversial contents avoid executing the algorithm
        controversial_contents = self.controversial_contents(alpha)
        if len(controversial_contents) == 0:
            return 0, [], 0

        # best score and corresponding users along iterations
        max_score = -1
        max_users_index = []
        max_n_nc_threads = 0

        while len(vertices_index) > 1:

            score_current, nc_threads = self.score_from_vertices_index(
                vertices_index, alpha, controversial_contents
            )
            n_nc_threads = len(nc_threads)

            if score_current > max_score:
                max_score = score_current
                max_users_index = vertices_index.copy()
                max_n_nc_threads = n_nc_threads

            # remove the node to obtain the highest score
            vertex_worst, new_score = self.__find_worst_vertex__(
                vertices_index, alpha, controversial_contents
            )

            # index of the worst node in the `vertices_index` array
            vertex_worst_index = vertices_index.index(vertex_worst)
            # remove the node from the array
            vertices_index.pop(vertex_worst_index)

        return max_score, max_users_index, max_n_nc_threads

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
            np.array: the parameters of a categorical distribution, based on the
        fraction of negative edges of the vertex
        """
        vertices_positiveness = self.vertices_positiveness()
        total_positiveness = np.sum(vertices_positiveness)

        return vertices_positiveness / total_positiveness

    def score_mip(
        self, alpha: float, relaxation: bool = False
    ) -> (int, list[int], list[tuple], list[int]):
        variables_cat = pulp.LpContinuous if relaxation else pulp.LpBinary
        variables_lb = 0 if relaxation else None
        variables_ub = 1 if relaxation else None

        controversial_contents = self.controversial_contents(alpha)
        if len(controversial_contents) == 0:
            # no controversial content and so no edge to be considered
            return 0, [], [], 0

        model = pulp.LpProblem("echo-chamber-score", pulp.LpMaximize)
        vertices_variables = [
            pulp.LpVariable(
                f"y_{index}",
                cat=variables_cat,
                lowBound=variables_lb,
                upBound=variables_ub,
            )
            for index in self.graph.get_vertices()
        ]

        # thread: ([negative edges vars],[edges variables]) dictionary
        thread_edges_dict = {}
        # dictionary of edge variables incident to edges
        vertices_edge_variables = {}
        # objective function of the problem
        objective = 0

        # thread: thread_variable (z_k)
        thread_k_vars = {}
        # list of all the edge variables x_ij
        edge_variables = []

        for i, edge in enumerate(self.graph.edges()):
            thread_obj = self.threads[edge]
            content = thread_obj.content

            # ignore non controversial contents
            if content in controversial_contents:
                source, target = tuple(edge)
                source = int(source)
                target = int(target)

                thread = thread_obj.url
                edge_var = pulp.LpVariable(
                    f"x_{source}_{target}_{i}",
                    lowBound=0,
                    upBound=1,
                )
                edge_variables.append(edge_var)
                # create the variable associated to this thread if it does not exist
                z_k = thread_k_vars.get(thread)
                if z_k is None:
                    z_k = pulp.LpVariable(
                        f"z_{hash(thread)}", lowBound=0, upBound=1
                    )
                    thread_k_vars[thread] = z_k

                objective += edge_var

                model += edge_var <= vertices_variables[source]
                model += edge_var <= vertices_variables[target]
                model += edge_var <= z_k
                model += (
                    edge_var
                    >= -2
                    + vertices_variables[source]
                    + vertices_variables[target]
                    + z_k
                )

                edges_negative_var, edges_var = thread_edges_dict.get(
                    thread, ([], [])
                )

                if self.weights[edge] < 0:
                    edges_negative_var.append(edge_var)
                edges_var.append(edge_var)

                thread_edges_dict[thread] = (edges_negative_var, edges_var)

                # add edge among edge variables of source
                source_edge_variables = vertices_edge_variables.get(source, [])
                source_edge_variables.append(edge_var)
                vertices_edge_variables[source] = source_edge_variables

                # add edge among edge variables of target
                target_edge_variables = vertices_edge_variables.get(target, [])
                target_edge_variables.append(edge_var)
                vertices_edge_variables[target] = target_edge_variables

        # add thread controversy constraints
        for k, edges_var_tuple in enumerate(thread_edges_dict.values()):
            edges_negative_var, edges_var = edges_var_tuple

            # sum of variables associated to negative edges of a single thread
            neg_edges_sum = pulp.lpSum(edges_negative_var)
            # sum of variables associated to edges of a single thread
            edges_sum = pulp.lpSum(edges_var)

            model += neg_edges_sum - alpha * edges_sum <= 0

        # add constraint for setting to one only vertices where at least one
        # edge is at 1
        for (
            vertex_index,
            vertex_edge_variables,
        ) in vertices_edge_variables.items():
            model += vertices_variables[vertex_index] <= pulp.lpSum(
                vertex_edge_variables
            )

        model += objective
        model.solve()

        score = pulp.value(model.objective)

        if score == 0:
            return 0, [], [], 0

        users = []
        for i, vertex_variable in enumerate(vertices_variables):
            if relaxation:
                # if relaxation problem, return value of all the vertices
                # instead of indices of non-zero nodes
                users.append(pulp.value(vertex_variable))
            elif pulp.value(vertex_variable) == 1:
                users.append(i)

        edges = []
        for i, edge_variable in enumerate(edge_variables):
            edge_name = edge_variable.name
            edge_name_split = edge_name.split("_")
            source = int(edge_name_split[1])
            target = int(edge_name_split[2])

            if relaxation:
                edge = (source, target, pulp.value(edge_variable))
                # if relaxation problem, return value of all the vertices
                # instead of indices of non-zero edges
                edges.append(edge)
            elif pulp.value(edge_variable) == 1:
                edge = (source, target, pulp.value(edge_variable))
                edges.append(edge)

        nc_threads = []
        for i, thread_variable in enumerate(thread_k_vars.values()):

            thread_value = pulp.value(thread_variable)
            if relaxation:
                # if relaxation problem, return value of all the vertices
                # instead of indices of non-zero threads
                nc_threads.append(thread_value)
            elif thread_value == 1:
                nc_threads.append(i)

        return score, users, edges, nc_threads

    def score_mip_densest(
        self, alpha: float, relaxation: bool = False
    ) -> (int, list[int], list[tuple], list[int]):
        variables_cat = pulp.LpContinuous if relaxation else pulp.LpBinary
        variables_lb = 0 if relaxation else None
        variables_ub = 1 if relaxation else None

        controversial_contents = self.controversial_contents(alpha)

        model = pulp.LpProblem("densest-echo-chamber-score", pulp.LpMaximize)
        vertices_binary_variables = [
            pulp.LpVariable(
                f"b_{index}",
                cat=variables_cat,
                lowBound=variables_lb,
                upBound=variables_ub,
            )
            for index in self.graph.get_vertices()
        ]
        vertices_continous_variables = [
            pulp.LpVariable(
                f"y_{index}",
                lowBound=0,
            )
            for index in self.graph.get_vertices()
        ]

        model += pulp.lpSum(vertices_continous_variables) <= 1
        # each continous variable must activate the corresponding binary one
        for vertex_continous_variable, vertex_binary_variable in zip(
            vertices_continous_variables, vertices_binary_variables
        ):
            model += vertex_continous_variable <= vertex_binary_variable

        # thread: ([negative edges vars],[edges variables]) dictionary
        thread_edges_dict = {}
        # objective function of the problem
        objective = 0

        # thread: thread_variable (z_k) dictionary
        thread_k_vars = {}
        # list of all the edge variables x_ij
        edge_variables = []

        for i, edge in enumerate(self.graph.edges()):
            thread_obj = self.threads[edge]
            content = thread_obj.content

            # ignore non controversial contents
            if content in controversial_contents:
                source, target = tuple(edge)
                source = int(source)
                target = int(target)

                thread = thread_obj.url
                edge_binary_var = pulp.LpVariable(
                    f"a_{source}_{target}_{i}",
                    cat=variables_cat,
                    lowBound=variables_lb,
                    upBound=variables_ub,
                )
                edge_continous_var = pulp.LpVariable(
                    f"x_{source}_{target}_{i}",
                    lowBound=0,
                )
                edge_variables.append(edge_continous_var)
                # create the variable associated to this thread if it does not exist
                z_k = thread_k_vars.get(thread)
                if z_k is None:
                    z_k = pulp.LpVariable(
                        f"z_{hash(thread)}", lowBound=0, upBound=1
                    )
                    thread_k_vars[thread] = z_k

                objective += edge_continous_var

                model += (
                    edge_continous_var <= vertices_continous_variables[source]
                )
                model += (
                    edge_continous_var <= vertices_continous_variables[target]
                )
                model += edge_continous_var <= edge_binary_var
                model += edge_binary_var <= z_k
                model += (
                    edge_binary_var
                    >= -2
                    + vertices_binary_variables[source]
                    + vertices_binary_variables[target]
                    + z_k
                )

                edges_negative_var, edges_var = thread_edges_dict.get(
                    thread, ([], [])
                )

                if self.weights[edge] < 0:
                    edges_negative_var.append(edge_binary_var)
                edges_var.append(edge_binary_var)

                thread_edges_dict[thread] = (edges_negative_var, edges_var)

        # add thread controversy constraints
        for k, edges_var_tuple in enumerate(thread_edges_dict.values()):
            edges_negative_var, edges_var = edges_var_tuple

            # sum of variables associated to negative edges of a single thread
            neg_edges_sum = pulp.lpSum(edges_negative_var)
            # sum of variables associated to edges of a single thread
            edges_sum = pulp.lpSum(edges_var)

            model += neg_edges_sum - alpha * edges_sum <= 0

        model += objective
        model.solve()

        score = pulp.value(model.objective)

        users = []
        for i, vertex_variable in enumerate(vertices_continous_variables):
            if relaxation:
                # if relaxation problem, return value of all the vertices
                # instead of indices of non-zero nodes
                users.append(pulp.value(vertex_variable))
            elif pulp.value(vertex_variable) > 0:
                users.append(i)

        edges = []
        for i, edge_variable in enumerate(edge_variables):
            edge_name = edge_variable.name
            edge_name_split = edge_name.split("_")
            source = int(edge_name_split[1])
            target = int(edge_name_split[2])

            if relaxation:
                edge = (source, target, pulp.value(edge_variable))
                # if relaxation problem, return value of all the vertices
                # instead of indices of non-zero edges
                edges.append(edge)
            elif pulp.value(edge_variable) > 0:
                edge = (source, target, pulp.value(edge_variable))
                edges.append(edge)

        #
        # Iterating over thread variables is not sufficient for detecting
        # active threads. In fact with this model formulation there can be some
        # variables set to 1 which do not actually contribute (this because
        # some vertices may have a 1 binary variable while still not taking any
        # of the available weight and so having the continuous variable set to
        # 0)
        #
        # nc_threads = []
        # for i, thread_variable in enumerate(thread_k_vars.values()):
        #
        #     thread_value = pulp.value(thread_variable)
        #     if relaxation:
        #         # if relaxation problem, return value of all the vertices
        #         # instead of indices of non-zero threads
        #         nc_threads.append(thread_value)
        #     elif thread_value == 1:
        #         nc_threads.append(i)

        _, nc_threads = self.score_from_vertices_index(users, alpha)

        return score, users, edges, nc_threads

    def score_relaxation_algorithm(
        self, alpha: float
    ) -> (int, list[int], int):
        controversial_contents = self.controversial_contents(alpha)

        score, users, edges, nc_threads = self.score_mip(
            alpha, relaxation=True
        )
        if score == 0:
            return score, users, nc_threads

        users = [user if user is not None else -1 for user in users]

        edges_np = np.array(edges)
        # exclude vertices set to 0
        edges_np = edges_np[edges_np[:, 2] != 0]

        # sort edges by weight in descending order
        edges_sorted = edges_np[np.flip(np.argsort(edges_np[:, 2]))]

        score_max = 0
        score_max_vertices = []
        score_max_n_nc_threads = 0

        vertices = set()
        vertices_size = 0

        for edge in edges_sorted:
            source = edge[0]
            target = edge[1]

            vertices.add(int(source))
            vertices.add(int(target))

            if len(vertices) != vertices_size:
                score, nc_threads = self.score_from_vertices_index(
                    vertices, alpha, controversial_contents
                )
                n_nc_threads = len(nc_threads)

                if score > score_max:
                    score_max = score
                    score_max_vertices = vertices.copy()
                    score_max_n_nc_threads = n_nc_threads

        return score_max, list(score_max_vertices), score_max_n_nc_threads

    def __aggregate_edges__(
        self,
        edges_ij: list[gt.Edge],
        alpha: float,
        controversial_contents: set,
        simple: bool,
    ):
        """Aggregate edges of a certain vertex

        Args:
            edges_ij (list[gt.Edge]): the list of edges between a pair of
            vertices
            alpha (float): alpha used for definying controversy
            controversial_contents (set): the list of controversial contents
            simple (bool): if True does not aggregate separately edges
            belonging to different threads

        Returns:
            the list of contents in which the edges will have a pair if simple
            is False, 1 or 0 otherwise (if there is an edge or not,
            respectively)
        """
        if simple:
            delta_minus_ij = 0
            delta_ij = 0

            for edge in edges_ij:
                edge_content = self.threads[edge].content

                if edge_content in controversial_contents:
                    edge_weight = self.weights[edge]

                    if edge_weight > 0:
                        delta_ij += edge_weight
                    else:
                        delta_ij -= edge_weight
                        delta_minus_ij -= edge_weight

            if delta_ij > 0 and delta_minus_ij / delta_ij <= alpha:
                return 1
            else:
                return 0
        else:
            # deltas for each thread
            thread_deltas_ij = {}

            for edge in edges_ij:
                edge_content = self.threads[edge].content

                if edge_content in controversial_contents:
                    edge_weight = self.weights[edge]
                    edge_thread = self.threads[edge].url

                    (
                        delta_minus_ij,
                        delta_ij,
                    ) = thread_deltas_ij.get(edge_thread, (0, 0))

                    if edge_weight > 0:
                        delta_ij += edge_weight
                    else:
                        delta_ij -= edge_weight
                        delta_minus_ij -= edge_weight

                    thread_deltas_ij[edge_thread] = (delta_minus_ij, delta_ij)

            threads = []
            for thread, delta_tuple in thread_deltas_ij.items():
                delta_minus_ij, delta_ij = delta_tuple

                if delta_minus_ij / delta_ij <= alpha:
                    threads.append(thread)

            return threads

    def nc_graph(
        self, alpha: float, simple: bool = True, layer: bool = False
    ) -> list[int]:
        controversial_contents = self.controversial_contents(alpha)

        # edges of the G_d graph
        edges = []

        for vertex_i in self.graph.vertices():
            i = int(vertex_i)

            for vertex_j in self.graph.vertices():
                j = int(vertex_j)

                if j > i:
                    edges_ij = self.graph.edge(
                        vertex_i, vertex_j, all_edges=True
                    )

                    edges_aggregated = self.__aggregate_edges__(
                        edges_ij, alpha, controversial_contents, simple
                    )

                    if simple:
                        if edges_aggregated > 0:
                            edges.append([i, j, 1])
                    elif layer:
                        edges.extend(
                            [[i, j, thread] for thread in edges_aggregated]
                        )
                    else:
                        # layer is false and simple is false
                        n_edges_aggregated = len(edges_aggregated)
                        if n_edges_aggregated > 0:
                            edges.append([i, j, n_edges_aggregated])

        num_vertices = int(vertex_i) + 1
        return num_vertices, edges

    def score_densest_nc_subgraph(
        self, alpha: float, simple: bool = True
    ) -> (float, list[int]):
        num_vertices, edges = self.nc_graph(alpha, simple, False)
        return densest.densest_subgraph(num_vertices, edges)

    def o2_bff_dcs_am(self, alpha: float, k: int) -> (int, list[int]):
        num_vertices, edges = self.nc_graph(alpha, False, layer=True)

        # construct the graph with the given vertices and edges
        graph = gt.Graph()
        vertices = list(graph.add_vertex(num_vertices))

        # create the content edge property for the graph
        content_property = graph.new_edge_property("string")
        graph.ep["content"] = content_property

        for edge in edges:
            edge_desc = graph.add_edge(vertices[edge[0]], vertices[edge[1]])
            content_property[edge_desc] = edge[2]

        return densest.o2_bff_dcs_am_incremental_overlap(graph, k)

    def select_echo_chamber(
        self,
        alpha: float,
        vertices_index: list[int] = None,
        controversial_contents: set = None,
    ):
        if vertices_index is None:
            _, users, _, _ = self.score_mip(alpha, controversial_contents)

        edge_filter = self.graph.new_edge_property("bool", val=False)
        vertex_filter = self.graph.new_vertex_property("bool", val=False)

        _, nc_threads = self.score_from_vertices_index(
            vertices_index, alpha, controversial_contents
        )

        # use a set for faster search
        vertices_index = set(vertices_index)
        nc_threads = set(nc_threads)

        for vertex_index in vertices_index:
            vertex = self.graph.vertex(vertex_index)
            vertex_filter[vertex] = True

            for edge in vertex.out_edges():
                # check if both the thread is non controversial and the target
                # node is in the echo chamber
                if (
                    self.threads[edge].url in nc_threads
                    and int(edge.target()) in vertices_index
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

    def is_induced_edge(self, vertices: set):
        is_induced_property = self.graph.new_edge_property("bool")
        for i, edge in enumerate(self.graph.get_edges()):
            if edge[0] in vertices or edge[1] in vertices:
                is_induced_property.a[i] = True

        return is_induced_property

    def clustering_accuracy(
        self,
        vertices_assignment: list[int],
        n_clusters: int,
        alpha: float,
        approximation: bool = True,
    ):
        current_edge_filter = self.graph.new_edge_property("bool")

        # array containing prediction of group for each vertex
        vertices_predicted = np.empty((self.graph.num_vertices()))
        vertices_predicted[:] = -1

        vertices_assignment = np.array(vertices_assignment)

        iterations_score = []
        for i in range(n_clusters):
            if approximation:
                _, vertices, _ = self.score_relaxation_algorithm(alpha)
            else:
                _, vertices, _, _ = self.score_mip(alpha)

            vertices_predicted[vertices] = i

            induced_edges_property = self.is_induced_edge(set(vertices))

            current_edge_filter.a = np.logical_or(
                current_edge_filter.a, induced_edges_property.a
            )
            self.graph.set_edge_filter(current_edge_filter, True)

            # compute jaccard coefficient for the current classification
            subgraph_vertices_assignment = vertices_assignment[vertices]
            majority_class = np.bincount(subgraph_vertices_assignment).argmax()

            class_assignment = (vertices_assignment == majority_class).astype(
                np.int32
            )
            class_prediction = np.zeros_like(class_assignment)
            class_prediction[vertices] = 1
            iteration_score = metrics.jaccard_score(
                class_assignment, class_prediction
            )
            iterations_score.append(iteration_score)

        self.clear_filters()

        adjusted_rand_score = metrics.adjusted_rand_score(
            vertices_assignment, vertices_predicted
        )
        rand_score = metrics.rand_score(
            vertices_assignment, vertices_predicted
        )

        jaccard_score = metrics.jaccard_score(
            vertices_assignment, vertices_predicted, average="micro"
        )
        return adjusted_rand_score, rand_score, jaccard_score, iterations_score

    def clear_filters(self):
        self.graph.clear_filters()
        return

    def shuffle(self):
        pass

    def label_nodes(self):
        twitter = TwitterCollector()

        for vertex in self.graph.vertices():
            vertex_screen_name = self.screen_names[vertex]

            label = twitter.get_user_label(vertex_screen_name)
            self.labels[vertex] = label

        return

    def deselect_unlabeled(self):
        vertex_filter = self.graph.new_vertex_property("bool")

        for vertex in self.graph.vertices():
            if self.labels[vertex] != -1:
                vertex_filter[vertex] = 1

        current_filter, _ = self.graph.get_vertex_filter()

        if current_filter is not None:
            vertex_filter.a = np.logical_and(current_filter.a, vertex_filter.a)

        self.graph.set_vertex_filter(vertex_filter)
        return

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
        """Creates a PolarizationGraph object from the given model parameters

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
        """Creates a PolarizationGraph object from the given model parameters

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
        """Creates a PolarizationGraph object from the graph stored in a file

        Args:
            filename (str): filename of the file where the graph is stored
        """
        graph = cls([])
        graph.load(filename)

        return graph
