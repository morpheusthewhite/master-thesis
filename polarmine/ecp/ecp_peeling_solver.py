from typing import List

from polarmine.ecp.ecp_solver import ECPSolver
from polarmine.ecp.utils import (
    score_from_vertices_index,
    __find_worst_vertex__,
)
from polarmine.graph import InteractionGraph


class ECPPeelingSolver(ECPSolver):
    """Solve the ECP through the peeling algorithm"""

    def solve(
        self, graph: InteractionGraph, alpha: float
    ) -> tuple[float, List[int], List[int], List[str]]:
        """Calculate the echo chamber score using the "peeling" greedy approach

        Args:
            alpha (float): maximum fraction of edges of non controversial content
        """
        vertices_index = list(graph.graph.get_vertices())

        # if there are no controversial contents avoid executing the algorithm
        controversial_contents = graph.controversial_contents(alpha)
        if len(controversial_contents) == 0:
            return 0, [], [], []

        # best score and corresponding users along iterations
        max_score: float = -1.0
        max_users_index: List[int] = []
        max_nc_threads: List[str] = []

        while len(vertices_index) > 1:

            score_current, nc_threads = score_from_vertices_index(
                graph, vertices_index, alpha, controversial_contents
            )

            if score_current > max_score:
                max_score = score_current
                max_users_index = vertices_index.copy()
                max_nc_threads = nc_threads

            # remove the node to obtain the highest score
            vertex_worst, _ = __find_worst_vertex__(
                graph, vertices_index, alpha, controversial_contents
            )

            # index of the worst node in the `vertices_index` array
            vertex_worst_index = vertices_index.index(vertex_worst)
            # remove the node from the array
            vertices_index.pop(vertex_worst_index)

        return max_score, max_users_index, [], max_nc_threads
