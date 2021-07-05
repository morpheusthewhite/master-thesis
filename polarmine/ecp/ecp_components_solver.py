from typing import List
import numpy as np
import graph_tool.all as gt

from polarmine.ecp.ecp_solver import ECPSolver
from polarmine.ecp.utils import score_from_vertices_index
from polarmine.graph import PolarizationGraph


class ECPComponentsSolver(ECPSolver):
    """Find solution for the ECP by considering each connected component as
    possible solution"""

    def solve(
        self, graph: PolarizationGraph, alpha: float
    ) -> tuple[float, List[int], List[int], List[str]]:
        component_label, _ = gt.label_components(graph.graph, directed=False)
        controversial_contents = graph.controversial_contents(alpha)

        n_components = int(np.max(component_label.a) + 1)
        max_score: float = 0.0
        max_users_index: List[int] = []
        max_nc_threads: List[str] = []

        for i in range(n_components):
            vertices_index = np.where(component_label.a == i)[0]

            score, nc_threads = score_from_vertices_index(
                graph, vertices_index, alpha, controversial_contents
            )

            if score > max_score:
                max_score = score
                max_users_index = vertices_index
                max_nc_threads = nc_threads

        return max_score, max_users_index, [], max_nc_threads
