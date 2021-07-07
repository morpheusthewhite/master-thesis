from typing import List

from polarmine.alternative.alternative_solver import AlternativeSolver
from polarmine.alternative.utils import densest_subgraph, nc_graph
from polarmine.graph import InteractionGraph


class PASolver(AlternativeSolver):
    """Solve the Densest Subgraph problem on the Pair Aggregated graph"""

    def solve(
        self, graph: InteractionGraph, alpha: float
    ) -> tuple[float, List[int]]:
        num_vertices, edges = nc_graph(graph, alpha, True, False)
        return densest_subgraph(num_vertices, edges)
