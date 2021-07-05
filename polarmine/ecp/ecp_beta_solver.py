from typing import List
import numpy as np

from polarmine.ecp.ecp_solver import ECPSolver
from polarmine.ecp.utils import (
    score_from_vertices_index,
    __find_best_neighbour__,
    __neighbours_merge__,
    __find_worst_vertex__,
    __neighbours_subtract__,
)
from polarmine.graph import PolarizationGraph


class ECPBetaSolver(ECPSolver):
    """Solve the ECP through the beta algorithm"""

    def __init__(
        self,
        beta: float = 1,
        n_starts: int = -1,
        positiveness_samples: bool = True,
        *args,
        **kwargs
    ):
        """Create a new ECPBetaSolver

        Args:
            beta (float): probability of adding a node along the iterations
            n_starts (int): number of times the algorithm is executed. If -1
            then it is sqrt(num_vertices)
            positiveness_samples (bool): if True, sample nodes according to
            their fraction of positive edges. Otherwise, sample uniformly
            the initial nodes
        """
        super(ECPBetaSolver, self).__init__(*args, **kwargs)
        self.beta = beta
        self.n_starts = n_starts
        self.positiveness_samples = positiveness_samples

    def solve(
        self, graph: PolarizationGraph, alpha: float
    ) -> tuple[float, List[int], List[int], List[str]]:
        """Calculate the echo chamber score using the beta greedy approach

        Args:
            alpha (float): maximum fraction of edges of non controversial content
        """
        vertices_index = graph.graph.get_vertices()

        if self.n_starts == -1:
            n_vertices = graph.num_vertices()
            self.n_starts = int(n_vertices ** (1 / 2))

        # if there are no controversial contents avoid executing the algorithm
        controversial_contents = graph.controversial_contents(alpha)
        if len(controversial_contents) == 0:
            return 0.0, [], [], []

        # best score and corresponding users along iterations
        score: float = -1
        users_index: List[int] = []
        nc_threads: List[str] = []

        if self.positiveness_samples:
            sample_probabilities = graph.positiveness_probabilities()

        for _ in range(self.n_starts):
            # list containing nodes which are temporarily ignored
            vertices_ignore = []

            if self.positiveness_samples:
                initial_vertex_index = np.where(
                    np.random.default_rng().multinomial(
                        1, sample_probabilities
                    )
                )[0][0]
            else:
                # sample a node, uniformly
                initial_vertex_index = np.random.default_rng().integers(
                    0, vertices_index.shape[0]
                )

            initial_vertex = vertices_index[initial_vertex_index]

            # current set of selected users
            vertices = [initial_vertex]
            # current set of neighbours of the selected users
            neighbours = set(graph.graph.get_all_neighbors(initial_vertex))

            score_current: float = -1.0
            # terminate the algorithm if no neighbour can be added
            while len(neighbours) > 0:

                # sample from a bernoulli to decide to add or not
                add_node = np.random.default_rng().binomial(1, self.beta)

                if add_node or len(vertices) == 1:
                    (
                        neighbour_best,
                        score_neighbour_best,
                    ) = __find_best_neighbour__(
                        graph,
                        vertices,
                        neighbours,
                        alpha,
                        controversial_contents,
                    )

                    # if no neighbour increases the score then stop
                    if score_neighbour_best <= score_current:
                        break
                    else:
                        vertices.append(neighbour_best)
                        neighbours.remove(neighbour_best)

                        # add to the list of neighbours the ones of the new node `neighbour`
                        __neighbours_merge__(
                            graph, neighbours, neighbour_best, vertices
                        )

                        score_current = score_neighbour_best
                else:
                    # remove the node contributing less to the score
                    (
                        vertex_worst,
                        score_vertex_worst,
                    ) = __find_worst_vertex__(
                        graph, vertices, alpha, controversial_contents
                    )

                    vertices.remove(vertex_worst)
                    # remove neighbours of the excluded node
                    __neighbours_subtract__(
                        graph, neighbours, vertex_worst, vertices
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
        score, nc_threads = score_from_vertices_index(
            graph, users_index, alpha, controversial_contents
        )
        return score, users_index, [], nc_threads
