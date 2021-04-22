import pulp


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
