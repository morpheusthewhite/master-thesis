from typing import List, Set, Iterable, Optional
import numpy as np
from polarmine.graph import InteractionGraph


def score_from_vertices_index(
    graph: InteractionGraph,
    vertices_index: Iterable[int],
    alpha: float,
    controversial_contents: Set[str] = None,
) -> tuple[float, List[str]]:
    """Calculate the Echo Chamber Score produced by a set of vertices

    Args:
        graph (InteractionGraph): the interaction graph
        vertices_index (list[int]): the indices of the vertices in the graph
        that are considered as solution
        alpha (float): alpha of Echo Chamber Score
        controversial_contents (set): the set of controversial contents in the
        graph

    Returns:
        tuple[float, List[str]]: the score and list of non controversial
        threads
    """
    thread_edges_dict = graph.__vertices_subthreads_dict__(
        vertices_index, alpha, controversial_contents
    )
    score: float = 0
    nc_threads: list[str] = []

    for thread, n_edges_tuple in thread_edges_dict.items():
        n_negative_edges, n_edges = n_edges_tuple

        if n_negative_edges / n_edges <= alpha:
            # non controversial threads
            score += n_edges - 2 * n_negative_edges
            nc_threads.append(thread)

    return score, nc_threads


def __neighbours_merge__(
    graph: InteractionGraph,
    neighbours: Set[int],
    vertex: int,
    vertices: Iterable[int],
):
    new_neighbours = graph.graph.get_all_neighbours(vertex)
    for new_neighbour in new_neighbours:
        if new_neighbour not in vertices:
            neighbours.add(new_neighbour)


def __neighbours_subtract__(
    graph: InteractionGraph,
    neighbours: Set[int],
    vertex_removed: int,
    vertices: Iterable[int],
):
    """remove from neighbours vertices which are no more reachable from vertices

    Args:
        neighbours (set(int)): a list of nodes
        vertex_removed (int): the vertex that has been removed
        vertices (list[int]): vertices of which neighbours should be kept
    """
    neighbours_removed = graph.graph.get_all_neighbours(vertex_removed)
    for neighbour_removed in neighbours_removed:

        # if this vertex is listed among `neighbours`, check if it is
        # reachable from another node
        if neighbour_removed in neighbours:

            reachable = False
            for vertex_i in graph.graph.get_all_neighbours(neighbour_removed):
                if vertex_i in vertices:
                    # it is reachable from another node
                    reachable = True
                    break

            if not reachable:
                neighbours.remove(neighbour_removed)
    return


def __find_worst_vertex__(
    graph: InteractionGraph,
    vertices: list[int],
    alpha: float,
    controversial_contents: Optional[Set[str]],
) -> tuple[int, float]:
    # keep neighbour increasing more the score
    vertices_worst = []
    score_vertex_worst = -1

    for i, vertex in enumerate(vertices):
        # vertices, excluding the current one
        vertices_current = vertices[:i] + vertices[i + 1 :]

        score_vertex, _ = score_from_vertices_index(
            graph,
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
    vertex_worst_index = np.random.default_rng().integers(
        0, len(vertices_worst)
    )
    vertex_worst = vertices_worst[vertex_worst_index]
    return vertex_worst, score_vertex_worst


def __find_best_neighbour__(
    graph: InteractionGraph,
    vertices: list[int],
    neighbours: Iterable[int],
    alpha: float,
    controversial_contents: Optional[Set[str]],
) -> tuple[int, float]:
    # keep neighbours increasing more the score
    neighbours_best = []
    score_neighbour_best = -1

    for neighbour in neighbours:
        score_neighbour, _ = score_from_vertices_index(
            graph,
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
    neighbour_best_index = np.random.default_rng().integers(
        0, len(neighbours_best)
    )
    neighbour_best = neighbours_best[neighbour_best_index]

    return neighbour_best, score_neighbour_best


def select_echo_chamber(
    graph: InteractionGraph,
    alpha: float,
    vertices_index: List[int],
    controversial_contents: Set[str] = None,
):

    edge_filter = graph.graph.new_edge_property("bool", val=False)
    vertex_filter = graph.graph.new_vertex_property("bool", val=False)

    nc_threads = []
    _, nc_threads = score_from_vertices_index(
        graph, vertices_index, alpha, controversial_contents
    )

    # use a set for faster search
    vertices_index_set = set(vertices_index)
    nc_threads = set(nc_threads)

    for vertex_index in vertices_index_set:
        vertex = graph.graph.vertex(vertex_index)
        vertex_filter[vertex] = True

        for edge in vertex.out_edges():
            # check if both the thread is non controversial and the target
            # node is in the echo chamber
            if (
                graph.threads[edge].url in nc_threads
                and int(edge.target()) in vertices_index_set
            ):
                edge_filter[edge] = True

    graph.graph.set_vertex_filter(vertex_filter)
    graph.graph.set_edge_filter(edge_filter)

    return
