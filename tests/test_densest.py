import graph_tool.all as gt

from polarmine.densest import densest_subgraph, dcs_am_exact


def test_densest_simple():
    num_vertices = 5
    edges = [
        [0, 1, 1],
        [4, 3, 1],
        [0, 3, 1],
        [0, 2, 1],
        [2, 3, 1],
        [2, 1, 1],
        [3, 1, 1],
    ]

    density, nodes = densest_subgraph(num_vertices, edges)

    assert density == 6 / 4

    assert 0 in nodes
    assert 1 in nodes
    assert 2 in nodes
    assert 3 in nodes
    assert 4 not in nodes


def test_dcs_am_exact1():
    graph = gt.Graph()
    contents_property = graph.new_edge_property("string")
    graph.ep["content"] = contents_property

    vertices = list(graph.add_vertex(5))

    edge = graph.add_edge(vertices[1], vertices[0])
    contents_property[edge] = "a"

    edge = graph.add_edge(vertices[2], vertices[1])
    contents_property[edge] = "a"

    edge = graph.add_edge(vertices[3], vertices[4])
    contents_property[edge] = "a"

    edge = graph.add_edge(vertices[3], vertices[0])
    contents_property[edge] = "b"

    edge = graph.add_edge(vertices[1], vertices[2])
    contents_property[edge] = "b"

    edge = graph.add_edge(vertices[4], vertices[2])
    contents_property[edge] = "b"

    score, vertices = dcs_am_exact(graph)
    print(vertices)
    assert score == 2


def test_dcs_am_exact2():
    graph = gt.Graph()
    contents_property = graph.new_edge_property("string")
    graph.ep["content"] = contents_property

    vertices = list(graph.add_vertex(5))

    edge = graph.add_edge(vertices[1], vertices[0])
    contents_property[edge] = "a"

    edge = graph.add_edge(vertices[2], vertices[1])
    contents_property[edge] = "a"

    edge = graph.add_edge(vertices[3], vertices[4])
    contents_property[edge] = "a"

    edge = graph.add_edge(vertices[3], vertices[0])
    contents_property[edge] = "b"

    edge = graph.add_edge(vertices[1], vertices[2])
    contents_property[edge] = "b"

    edge = graph.add_edge(vertices[4], vertices[2])
    contents_property[edge] = "b"

    edge = graph.add_edge(vertices[2], vertices[1])
    contents_property[edge] = "c"

    score, vertices = dcs_am_exact(graph)
    print(vertices)
    assert score == 3
    assert set(vertices) == {1, 2}
