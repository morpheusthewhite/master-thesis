from typing import List
import numpy as np
import graph_tool.all as gt

from polarmine.ecp.ecp_solver import ECPSolver
from polarmine.ecp.ecp_mip_solver import ECPMIPSolver
from polarmine.ecp.utils import score_from_vertices_index
from polarmine.graph import PolarizationGraph


class ECPRoundingSolver(ECPSolver):
    """Solve the ECP through the rounding algorithm"""

    def solve(
        self, graph: PolarizationGraph, alpha: float
    ) -> tuple[float, List[int], List[int], List[str]]:
        controversial_contents = graph.controversial_contents(alpha)

        score, users, edges, nc_threads = ECPMIPSolver(relaxation=True).solve(
            graph, alpha
        )
        if score == 0.0:
            return score, users, edges, nc_threads

        def score_from_components(
            components_decomposition: np.array, num_components: int
        ) -> tuple[float, List[int], List[str]]:
            max_score: float = -1.0
            max_score_vertices: List[int] = []
            max_nc_threads: List[str] = []

            for i in range(num_components):
                components_vertices = np.where(
                    components_decomposition.a == i
                )[0]

                if components_vertices.shape[0] > 1:
                    score, nc_threads = score_from_vertices_index(
                        graph, components_vertices, alpha
                    )

                    if score > max_score:
                        max_score = score
                        max_score_vertices = components_vertices
                        max_nc_threads = nc_threads

            return max_score, max_score_vertices, max_nc_threads

        users = [user if user is not None else -1 for user in users]

        edges_np = np.array(edges)
        # exclude vertices set to 0
        edges_np = edges_np[edges_np[:, 2] != 0]

        # sort edges by weight in descending order
        np.random.default_rng().shuffle(edges_np)
        edges_sorted = edges_np[np.flip(np.argsort(edges_np[:, 2]))]

        score_max: float = 0.0
        score_max_vertices: list[int] = []
        score_max_nc_threads: list[str] = []

        vertices = set()
        vertices_size = 0

        # graph used for detecting connected components
        components_graph = gt.Graph(directed=False)
        components_graph.set_fast_edge_removal(True)

        old_num_components = -1

        for edge in edges_sorted:
            source = edge[0]
            target = edge[1]

            vertices.add(int(source))
            vertices.add(int(target))

            components_graph.add_edge(source, target)

            components, _ = gt.label_components(components_graph)
            num_components = np.max(components.a) + 1

            if num_components != old_num_components:
                old_num_components = num_components
                (
                    score,
                    vertices_components,
                    nc_threads,
                ) = score_from_components(components, num_components)

                if score > score_max:
                    score_max = score
                    score_max_vertices = vertices_components
                    score_max_nc_threads = nc_threads

            if len(vertices) != vertices_size:
                score, nc_threads = score_from_vertices_index(
                    graph, vertices, alpha, controversial_contents
                )

                if score > score_max:
                    score_max = score
                    score_max_vertices = vertices.copy()
                    score_max_nc_threads = nc_threads

        return score_max, list(score_max_vertices), [], score_max_nc_threads
