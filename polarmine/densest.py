import itertools
import pulp
import graph_tool.all as gt
import numpy as np

from sklearn.metrics import jaccard_score


def densest_subgraph(
    num_vertices: int, edges: list[int]
) -> (float, list[int]):
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

    density = pulp.value(model.objective)

    vertices = []
    for i, vertex_variable in enumerate(vertices_variables):
        if pulp.value(vertex_variable) > 0:
            vertices.append(i)

    return density, vertices


def select_contents(graph: gt.Graph, contents: list[str]):
    contents = graph.edge_properties["content"]
    edge_filter = graph.new_edge_property("bool")

    for edge in graph.edges():
        edge_filter[edge] = contents[edge] in contents

    current_edge_filter, _ = graph.get_edge_filter()

    if current_edge_filter is not None:
        edge_filter.a = np.logical_and(edge_filter.a, current_edge_filter.a)

    graph.set_edge_filter(edge_filter)


def dcs_am_exact(graph: gt.Graph):
    """Compute exactly DCS-AM using Charikar's k-core algorithm

    Args:
        graph (gt.Graph): a graph with an inner edge property "content"
        associated to the content of the edges
    """

    def select_kcore(graph: gt.Graph, k: int):
        kcore = gt.kcore_decomposition(graph)
        vertex_filter = graph.new_vertex_property("bool")

        for vertex in graph.vertices():
            vertex_filter[vertex] = kcore[vertex] >= k

        graph.set_vertex_filter(vertex_filter)

    contents = set(graph.edge_properties["content"])

    def find_k_list_core(graph: gt.Graph, k_list: list[int]) -> np.array:
        for content, k in zip(contents, k_list):
            select_contents(graph, [content])
            select_kcore(graph, k)

            if graph.num_vertices() == 0:
                return []

            graph.set_edge_filter(None)

        return graph.get_vertices()

    score_best = 0
    vertices_best = []

    for k_list in itertools.product(
        range(graph.num_vertices()), repeat=len(contents)
    ):
        score = sum(k_list)

        # consider the current k_list only if it will produce a score higher
        # than the best one
        if score > score_best:
            vertices = find_k_list_core(graph, k_list)

            # the algorithm found a solution if the vertices set is not empty
            if len(vertices) > 0:
                score_best = score
                vertices_best = vertices

            graph.clear_filters()

    return score_best, vertices_best


def score_a(graph: gt.Graph, vertex: int, n_contents: int) -> float:
    degree = graph.get_all_edges(vertex).shape[0]

    return degree / n_contents


def dcs_am_from_vertices(graph: gt.Graph) -> int:
    """calculate DCS-AM score of the selected vertices in the graph

    Args:
        graph (gt.Graph): a graph where vertices not in S are filtered out

    Returns:
        int: the DCS-AM score
    """
    contents_property = graph.ep["content"]
    contents = set(graph.edge_properties["content"])
    contents_degree_dict = {}

    for vertex in graph.vertices():
        vertex_content_degree_dict = {}

        for edge in vertex.all_edges():
            edge_content = contents_property[edge]
            degree = vertex_content_degree_dict.get(edge_content, 0) + 1
            vertex_content_degree_dict[edge_content] = degree

        for content in contents:
            # if the content is not present then the degree of the vertex is 0
            vertex_degree = vertex_content_degree_dict.get(content, 0)

            # append the degree of the current vertex to the degree of the
            # other vertices
            degrees = contents_degree_dict.get(content, [])
            degrees.append(vertex_degree)

            contents_degree_dict[content] = degrees

    dcs_am_score = 0
    for content, degrees in contents_degree_dict.items():
        dcs_am_score += min(degrees)

    return dcs_am_score


def find_bff_a(graph: gt.Graph, num_contents: int) -> (int, list[int]):
    num_vertices = graph.num_vertices()

    def filter_vertex(vertex: int):
        vertex_filter, _ = graph.get_vertex_filter()

        if vertex_filter is None:
            vertex_filter = graph.new_vertex_property("bool", val=True)

        vertex_filter.a[vertex] = False
        graph.set_vertex_filter(vertex_filter)

    max_dcs_am_score = -1
    max_dcs_am_vertices = []

    for i in range(num_vertices):
        # initialize the min with 2 since its maximum possible value is 1 (1
        # edge per layer
        min_score = 2
        min_vertex = -1

        for vertex in graph.get_vertices():
            score = score_a(graph, vertex, num_contents)

            if score < min_score:
                min_score = score
                min_vertex = vertex

        filter_vertex(min_vertex)
        dcs_am_score = dcs_am_from_vertices(graph)

        if dcs_am_score > max_dcs_am_score:
            max_dcs_am_score = dcs_am_score
            max_dcs_am_vertices = graph.get_vertices()

    return max_dcs_am_score, max_dcs_am_vertices


def o2_bff_dcs_am_incremental_overlap(
    graph: gt.Graph, k: int
) -> (int, list[int]):
    contents = list(set(graph.edge_properties["content"]))

    # return a trivial solution if there are less than 2 contents
    # even if this is not numerically correct
    if len(contents) < 2:
        return 0, []

    num_vertices = graph.num_vertices()
    S_i = []

    for content in contents:
        select_contents(graph, [content])
        score, vertices = find_bff_a(graph, len(contents))

        vertices_i_bin = np.zeros(num_vertices)
        vertices_i_bin[vertices] = 1

        S_i.append(vertices_i_bin)
        graph.clear_filters()

    max_pair = ()
    max_jaccard_score = -1

    # find the pair of vertices which are most similar (jaccard score)
    for indices_pair in itertools.combinations(range(len(contents)), 2):
        i, j = indices_pair
        vertices_i, vertices_j = S_i[i], S_i[j]

        score = jaccard_score(vertices_i, vertices_j)
        if score > max_jaccard_score:
            max_jaccard_score = score
            max_pair = (i, j)

    # most similar pair of vertices: these are the corresponding indices
    i, j = max_pair
    C_prev = {contents[i], contents[j]}

    for _ in range(2, k):
        select_contents(graph, C_prev)
        _, vertices = find_bff_a(graph, len(contents))

        # array of binaries representing which vertices are in the current S
        vertices_bin = np.zeros(num_vertices)
        vertices_bin[vertices] = 1

        # variables needed to keep most similar new set of vertices
        max_content = -1
        max_jaccard_score = -1

        # choose which content to add to the previous ones
        for content_index, content in enumerate(contents):

            # consider only content which are not already selected
            if content not in C_prev:

                # compute the jaccard similarity between the S_i of this
                # content and the current set of vertices associated with
                # C_prev
                content_vertices = S_i[content_index]

                score = jaccard_score(content_vertices, vertices_bin)

                if score > max_jaccard_score:
                    max_jaccard_score = score
                    max_content = content_index

        C_prev.add(max_content)
        graph.clear_filters()

    select_contents(graph, C_prev)
    graph.clear_filters()

    return find_bff_a(graph, len(contents))
