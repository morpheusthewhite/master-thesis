from polarmine.densest import densest_subgraph


def test_densest_simple():
    num_vertices = 5
    edges = [
        [0, 1],
        [4, 3],
        [0, 3],
        [0, 2],
        [2, 3],
        [2, 1],
        [3, 1],
    ]

    density, nodes = densest_subgraph(num_vertices, edges)

    assert density == 6 / 4

    assert 0 in nodes
    assert 1 in nodes
    assert 2 in nodes
    assert 3 in nodes
    assert 4 not in nodes
