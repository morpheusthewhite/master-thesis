from typing import List, Iterable, Union
import pulp
import graph_tool.all as gt
import numpy as np

from polarmine.graph import InteractionGraph


def densest_subgraph(
    num_vertices: int, edges: List[List[int]]
) -> tuple[float, List[int]]:
    """Find densest subgraph by using Charikar's LP model

    Args:
        num_vertices (int): the number of vertices in the graph
        edges (list[int]): array of edges, each represented by a pair of
        indices (of the vertices) and the corresponding weight. Each index
        should be in {0, ..., num_vertices-1}

    Returns:
        (float, list[int]): the density and the indices of the vertices in the
        densest subgraph
    """
    model = pulp.LpProblem("densest-subgraph", pulp.LpMaximize)
    vertices_variables = [
        pulp.LpVariable(
            f"y_{index}",
            lowBound=0,
        )
        for index in range(num_vertices)
    ]

    objective = 0

    for i, edge in enumerate(edges):
        source = edge[0]
        target = edge[1]
        weight = edge[2]

        edge_var = pulp.LpVariable(
            f"x_{source}_{target}_{i}",
            lowBound=0,
        )

        objective += weight * edge_var

        model += edge_var <= vertices_variables[source]
        model += edge_var <= vertices_variables[target]

    model += pulp.lpSum(vertices_variables) == 1
    model += objective
    model.solve()

    density: float = pulp.value(model.objective)

    vertices: List[int] = []
    for i, vertex_variable in enumerate(vertices_variables):
        if pulp.value(vertex_variable) > 0:
            vertices.append(i)

    return density, vertices


def select_contents(graph: gt.Graph, contents: Iterable[str]):
    """Select edges of the graph associated to the passed contents

    Args:
        graph (gt.Graph): a graph_tool graph
        contents (List[str]): the list of contents
    """
    contents_property = graph.edge_properties["content"]
    edge_filter = graph.new_edge_property("bool")

    for edge in graph.edges():
        edge_filter[edge] = contents_property[edge] in contents

    current_edge_filter, _ = graph.get_edge_filter()

    if current_edge_filter is not None:
        edge_filter.a = np.logical_and(edge_filter.a, current_edge_filter.a)

    graph.set_edge_filter(edge_filter)


def nc_graph(
    graph: InteractionGraph,
    alpha: float,
    simple: bool = True,
    layer: bool = False,
) -> tuple[int, List[List[int]]]:
    controversial_contents = graph.controversial_contents(alpha)

    # edges of the G_d graph
    edges = []

    vertex_i = -1
    for vertex_i in graph.graph.vertices():
        i = int(vertex_i)

        for vertex_j in graph.graph.vertices():
            j = int(vertex_j)

            if j > i:
                edges_ij: list[gt.Edge] = graph.graph.edge(
                    vertex_i, vertex_j, all_edges=True
                )

                edges_aggregated = __aggregate_edges__(
                    graph, edges_ij, alpha, controversial_contents, simple
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


def __aggregate_edges__(
    graph: InteractionGraph,
    edges_ij: list[gt.Edge],
    alpha: float,
    controversial_contents: set,
    simple: bool,
) -> Union[List[str], int]:
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
            edge_content = graph.threads[edge].content

            if edge_content in controversial_contents:
                edge_weight = graph.weights[edge]

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
            edge_content = graph.threads[edge].content

            if edge_content in controversial_contents:
                edge_weight = graph.weights[edge]
                edge_thread = graph.threads[edge].url

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
