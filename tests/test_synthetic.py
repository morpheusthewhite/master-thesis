import time
import os
import numpy as np
import matplotlib.pyplot as plt

from polarmine.graph import PolarizationGraph

OUTDIR = os.path.join("out", "synthetic")


def test_synthetic(iterations: int = 1):
    if not os.path.exists(OUTDIR):
        dir1 = os.path.split(OUTDIR)[0]

        if not os.path.exists(dir1):
            os.mkdir(dir1)

        os.mkdir(OUTDIR)

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
    n_nodes = [40, 40]
    omega_positive = [
        [0.8, 0.1],
        [0.1, 0.8],
    ]
    omega_negative = [
        [0.1, 0.8],
        [0.8, 0.1],
    ]

    n_nodes_list.append(n_nodes)
    n_threads_list.append(10)
    omega_positive_list.append(np.array(omega_positive) / 32)
    omega_negative_list.append(np.array(omega_negative) / 32)

    # this second graph is much simpler. It has 40 vertices and 2
    # different echo chambers, one for each community
    n_nodes = [40, 40]
    omega_positive = [
        [0.9, 0.01],
        [0.01, 0.9],
    ]
    omega_negative = [
        [0.01, 0.9],
        [0.9, 0.01],
    ]

    n_nodes_list.append(n_nodes)
    n_threads_list.append(10)
    omega_positive_list.append(np.array(omega_positive) / 32)
    omega_negative_list.append(np.array(omega_negative) / 32)

    n_nodes = [20, 20]
    omega_positive = [
        [0.45, 0.45],
        [0.45, 0.45],
    ]
    omega_negative = [
        [0.45, 0.45],
        [0.45, 0.45],
    ]

    n_nodes_list.append(n_nodes)
    n_threads_list.append(10)
    omega_positive_list.append(np.array(omega_positive) / 32)
    omega_negative_list.append(np.array(omega_negative) / 32)

    n_nodes = [20, 20]
    omega_positive = [
        [0.20, 0.20],
        [0.20, 0.20],
    ]
    omega_negative = [
        [0.60, 0.60],
        [0.60, 0.60],
    ]

    n_nodes_list.append(n_nodes)
    n_threads_list.append(10)
    omega_positive_list.append(np.array(omega_positive) / 32)
    omega_negative_list.append(np.array(omega_negative) / 32)

    # keep track of iteration
    i = 0

    for n_nodes, n_threads, omega_positive, omega_negative in zip(
        n_nodes_list, n_threads_list, omega_positive_list, omega_negative_list
    ):
        omega_positive_pdf = os.path.join(OUTDIR, f"omega_positive{i}.pdf")
        plt.matshow(omega_positive / np.max(omega_positive))
        plt.savefig(omega_positive_pdf)

        omega_negative_pdf = os.path.join(OUTDIR, f"omega_negative{i}.pdf")
        plt.matshow(omega_negative / np.max(omega_negative))
        plt.savefig(omega_negative_pdf)

        scores = np.empty((iterations,))
        for k in range(iterations):
            # generate a graph
            graph = PolarizationGraph.from_model(
                n_nodes, n_threads, omega_positive, omega_negative
            )

            start = time.time()
            score, _, _ = graph.score_relaxation_algorithm(0.2)
            scores[k] = score
            end = time.time()

        # create the array encoding the communities from the number of nodes
        # will save to file only one of the graphs
        communities = []
        for j, n_group_nodes in enumerate(n_nodes):
            communities += [j] * n_group_nodes

        outfile = os.path.join(OUTDIR, f"graph{i}.pdf")
        graph.draw(output=outfile, communities=communities)

        print("-" * 30)
        print(f"Vertices: {graph.num_vertices()}; Edges: {graph.num_edges()}")
        print(f"Omega positive: {omega_positive}")
        print(f"Omega negative: {omega_negative}")
        print(f"Score MIP: {np.average(scores)}")
        print(f"Time: {end - start}")
        print("-" * 30)

        i += 1


if __name__ == "__main__":
    test_synthetic(5)
