from typing import List

from polarmine.alternative.alternative_solver import AlternativeSolver
from polarmine.alternative.utils import densest_subgraph, nc_graph
from polarmine.graph import InteractionGraph


class TPADensestSolver(AlternativeSolver):
    """Solve the Densest Subgraph problem on the Thread Pair Aggregated graph"""

    def solve(
        self, graph: InteractionGraph, alpha: float
    ) -> tuple[float, List[int]]:
        num_vertices, edges = nc_graph(graph, alpha, False, False)
        return densest_subgraph(num_vertices, edges)
