import graph_tool.all as gt

from polarmine import densest


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

    density, nodes = densest.densest_subgraph(num_vertices, edges)

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

    score, vertices = densest.dcs_am_exact(graph)
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

    score, vertices = densest.dcs_am_exact(graph)
    assert score == 3
    assert set(vertices) == {1, 2}


def test_dcs_am_score():
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

    score = densest.dcs_am_from_vertices(graph)
    assert score == 2

    edge = graph.add_edge(vertices[2], vertices[1])
    contents_property[edge] = "c"

    score = densest.dcs_am_from_vertices(graph)
    assert score == 2

    filter_property = graph.new_vertex_property("bool")
    filter_property.a[1] = True
    filter_property.a[2] = True
    graph.set_vertex_filter(filter_property)

    score = densest.dcs_am_from_vertices(graph)
    assert score == 3


def test_find_bff_m():
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

    score1, vertices = densest.find_bff_m(graph, 3)

    # verity that the returned set produces the given score
    filter_property = graph.new_vertex_property("bool")
    filter_property.a[list(vertices)] = True
    graph.set_vertex_filter(filter_property)

    score2 = densest.dcs_am_from_vertices(graph)

    assert score1 == score2


def test_o2_bff():
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

    score1, vertices = densest.o2_bff_dcs_am_incremental_overlap(graph, 2)

    # verity that the returned set produces the given score
    filter_property = graph.new_vertex_property("bool")
    filter_property.a[list(vertices)] = True
    graph.set_vertex_filter(filter_property)

    score2 = densest.dcs_am_from_vertices(graph)

    assert score1 == score2
