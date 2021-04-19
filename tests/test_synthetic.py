import time
import numpy as np
import matplotlib.pyplot as plt
import graph_tool.spectral as gt

from polarmine.graph import PolarizationGraph


def test_synthetic():
    n_nodes_list = []
    n_threads_list = []
    omega_positive_list = []
    omega_negative_list = []

    # this first graph is not sparse at all. It has 100 vertices and it has
    # different echo chambers, approximately one for each community
    n_nodes = [40, 20, 30, 10]
    omega_positive = [
        [0.3, 0.05, 0.1, 0.05],
        [0.05, 0.35, 0.06, 0.05],
        [0.06, 0.05, 0.30, 0.05],
        [0.05, 0.05, 0.04, 0.30],
    ]
    omega_negative = [
        [0.2, 0.4, 0.22, 0.3],
        [0.4, 0.2, 0.4, 0.3],
        [0.5, 0.4, 0.2, 0.6],
        [0.4, 0.33, 0.3, 0.2],
    ]

    n_nodes_list.append(n_nodes)
    n_threads_list.append(2)
    omega_positive_list.append(np.array(omega_positive) / 4)
    omega_negative_list.append(np.array(omega_negative) / 16)

    # this second graph is much simpler. It has 40 vertices and 2
    # different echo chambers, one for each community
    n_nodes = [30, 30]
    omega_positive = [
        [0.8, 0.1],
        [0.1, 0.8],
    ]
    omega_negative = [
        [0.1, 0.8],
        [0.8, 0.1],
    ]

    n_nodes_list.append(n_nodes)
    n_threads_list.append(2)
    omega_positive_list.append(np.array(omega_positive) / 2)
    omega_negative_list.append(np.array(omega_negative) / 2)

    n_nodes = [30, 30]
    omega_positive = [
        [0.45, 0.45],
        [0.45, 0.45],
    ]
    omega_negative = [
        [0.45, 0.45],
        [0.45, 0.45],
    ]

    n_nodes_list.append(n_nodes)
    n_threads_list.append(2)
    omega_positive_list.append(np.array(omega_positive) / 2)
    omega_negative_list.append(np.array(omega_negative) / 2)

    for n_nodes, n_threads, omega_positive, omega_negative in zip(
        n_nodes_list, n_threads_list, omega_positive_list, omega_negative_list
    ):
        # generate a graph
        graph = PolarizationGraph.from_model(
            n_nodes, n_threads, omega_positive, omega_negative
        )

        assert graph.num_vertices() == np.sum(n_nodes)

        # adjacency = gt.adjacency(graph.graph)
        # print(adjacency)
        #
        # plt.figure()
        # plt.matshow(adjacency)
        # plt.show()

        # create the array encoding the communities from the number of nodes
        communities = []
        for i, n_group_nodes in enumerate(n_nodes):
            communities += [i] * n_group_nodes

        graph.draw(communities=communities)

        start = time.time()
        score, _, _, _ = graph.score_mip(0.2)
        end = time.time()

        print("-" * 30)
        print(f"Vertices: {graph.num_vertices()}; Edges: {graph.num_edges()}")
        print(f"Score MIP: {score}")
        print(f"Time: {end - start}")
        print("-" * 30)

    assert False


test_synthetic()
