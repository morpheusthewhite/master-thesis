import time
import os
import numpy as np
import matplotlib.pyplot as plt

from polarmine.graph import PolarizationGraph

OUTDIR = os.path.join("out", "synthetic")


def evaluate_graph(
    graph: PolarizationGraph,
    alpha: float,
    n_communities: int,
    communities: list[int],
):

    start = time.time()
    score, _, _ = graph.score_relaxation_algorithm(alpha)

    adjusted_rand_score, rand_score = graph.clustering_accuracy(
        communities, n_communities, alpha
    )
    end = time.time()

    return score, rand_score, adjusted_rand_score, end - start


def print_results(
    graph: PolarizationGraph,
    omega_positive: np.array,
    omega_negative: np.array,
    scores: np.array,
    duration: int,
    rand_scores: np.array,
    adjusted_rand_scores: np.array,
):
    # Will save to file only one of the graphs
    print("-" * 30)
    print(f"Vertices: {graph.num_vertices()}; Edges: {graph.num_edges()}")
    print(f"Omega positive: {omega_positive}")
    print(f"Omega negative: {omega_negative}")
    print(f"Fraction of negative edges: {graph.negative_edges_fraction()}")
    print(f"Score MIP: {np.average(scores)}")
    print(f"Time: {duration}")
    print(f"Clustering Rand score: {np.average(rand_scores)}")
    print(
        f"Clustering Adjusted Rand score: {np.average(adjusted_rand_scores)}"
    )
    print("-" * 30)

    return


def test_synthetic1(iterations: int = 1):
    n_nodes_list = []
    n_threads_list = []
    omega_positive_list = []
    omega_negative_list = []
    n_active_communities_list = []
    theta_list = []

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
    n_active_communities_list.append(2)
    theta_list.append(1)

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
    n_active_communities_list.append(2)
    theta_list.append(1)

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
    n_active_communities_list.append(2)
    theta_list.append(1)

    n_nodes = [40, 40]
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
    n_active_communities_list.append(2)
    theta_list.append(1)

    n_nodes = [40, 40]
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
    n_active_communities_list.append(2)
    theta_list.append(1)

    # keep track of iteration
    i = 0
    alpha = 0.2

    scores = np.empty((iterations,))
    rand_scores = np.empty((iterations,))
    adjusted_rand_scores = np.empty((iterations,))

    for (
        n_nodes,
        n_threads,
        omega_positive,
        omega_negative,
        n_active_communities,
        theta,
    ) in zip(
        n_nodes_list,
        n_threads_list,
        omega_positive_list,
        omega_negative_list,
        n_active_communities_list,
        theta_list,
    ):
        n_communities = len(omega_positive)

        omega_positive_pdf = os.path.join(
            OUTDIR, f"model1_omega_positive{i}.pdf"
        )
        plt.matshow(omega_positive / np.max(omega_positive))
        plt.savefig(omega_positive_pdf)

        omega_negative_pdf = os.path.join(
            OUTDIR, f"model1_omega_negative{i}.pdf"
        )
        plt.matshow(omega_negative / np.max(omega_negative))
        plt.savefig(omega_negative_pdf)

        for k in range(iterations):
            # generate a graph
            graph = PolarizationGraph.from_model1(
                n_nodes,
                n_threads,
                omega_positive,
                omega_negative,
                n_active_communities,
                theta,
            )
            # create the array encoding the communities from the number of
            # nodes.
            communities = []
            for j, n_group_nodes in enumerate(n_nodes):
                communities += [j] * n_group_nodes

            score, rand_score, adjusted_rand_score, duration = evaluate_graph(
                graph, alpha, n_communities, communities
            )

            scores[k] = score
            rand_scores[k] = rand_score
            adjusted_rand_scores[k] = adjusted_rand_score

        outfile = os.path.join(OUTDIR, f"model1_graph{i}.pdf")
        graph.draw(output=outfile, communities=communities)

        print_results(
            graph,
            omega_positive,
            omega_negative,
            scores,
            duration,
            rand_scores,
            adjusted_rand_scores,
        )

        i += 1


def test_synthetic2(iterations: int = 1):

    n_nodes_list = []
    n_threads_list = []
    omega_positive_list = []
    omega_negative_list = []
    phi_list = []
    n_active_communities_list = []
    theta_list = []
    beta_a_list = []
    beta_n_list = []

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
    phi = [
        [0.3, 0.05, 0.1, 0.05],
        [0.05, 0.35, 0.06, 0.05],
        [0.06, 0.05, 0.30, 0.05],
        [0.05, 0.05, 0.04, 0.30],
    ]

    n_nodes_list.append(n_nodes)
    n_threads_list.append(2)
    omega_positive_list.append(np.array(omega_positive) / 4)
    omega_negative_list.append(np.array(omega_negative) / 16)
    phi_list.append(np.array(phi))
    n_active_communities_list.append(2)
    theta_list.append(1)
    beta_a_list.append(0.04)
    beta_n_list.append(0.04)

    #     # this second graph is much simpler. It has 40 vertices and 2
    #     # different echo chambers, one for each community
    #     n_nodes = [40, 40]
    #     omega_positive = [
    #         [0.8, 0.1],
    #         [0.1, 0.8],
    #     ]
    #     omega_negative = [
    #         [0.1, 0.8],
    #         [0.8, 0.1],
    #     ]
    #
    #     n_nodes_list.append(n_nodes)
    #     n_threads_list.append(10)
    #     omega_positive_list.append(np.array(omega_positive) / 32)
    #     omega_negative_list.append(np.array(omega_negative) / 32)
    #     n_active_communities_list.append(2)
    #     theta_list.append(1)
    #
    #     # this second graph is much simpler. It has 40 vertices and 2
    #     # different echo chambers, one for each community
    #     n_nodes = [40, 40]
    #     omega_positive = [
    #         [0.9, 0.01],
    #         [0.01, 0.9],
    #     ]
    #     omega_negative = [
    #         [0.01, 0.9],
    #         [0.9, 0.01],
    #     ]
    #
    #     n_nodes_list.append(n_nodes)
    #     n_threads_list.append(10)
    #     omega_positive_list.append(np.array(omega_positive) / 32)
    #     omega_negative_list.append(np.array(omega_negative) / 32)
    #     n_active_communities_list.append(2)
    #     theta_list.append(1)
    #
    #     n_nodes = [40, 40]
    #     omega_positive = [
    #         [0.45, 0.45],
    #         [0.45, 0.45],
    #     ]
    #     omega_negative = [
    #         [0.45, 0.45],
    #         [0.45, 0.45],
    #     ]
    #
    #     n_nodes_list.append(n_nodes)
    #     n_threads_list.append(10)
    #     omega_positive_list.append(np.array(omega_positive) / 32)
    #     omega_negative_list.append(np.array(omega_negative) / 32)
    #     n_active_communities_list.append(2)
    #     theta_list.append(1)
    #
    #     n_nodes = [40, 40]
    #     omega_positive = [
    #         [0.20, 0.20],
    #         [0.20, 0.20],
    #     ]
    #     omega_negative = [
    #         [0.60, 0.60],
    #         [0.60, 0.60],
    #     ]
    #
    #     n_nodes_list.append(n_nodes)
    #     n_threads_list.append(10)
    #     omega_positive_list.append(np.array(omega_positive) / 32)
    #     omega_negative_list.append(np.array(omega_negative) / 32)
    #     n_active_communities_list.append(2)
    #     theta_list.append(1)
    #
    # keep track of iteration
    i = 0
    alpha = 0.2

    for (
        n_nodes,
        n_threads,
        omega_positive,
        omega_negative,
        phi,
        n_active_communities,
        theta,
        beta_a,
        beta_n,
    ) in zip(
        n_nodes_list,
        n_threads_list,
        omega_positive_list,
        omega_negative_list,
        phi_list,
        n_active_communities_list,
        theta_list,
        beta_a_list,
        beta_n_list,
    ):
        n_communities = len(omega_negative)

        omega_positive_pdf = os.path.join(
            OUTDIR, f"model2_omega_positive{i}.pdf"
        )
        plt.matshow(omega_positive / np.max(omega_positive))
        plt.savefig(omega_positive_pdf)

        omega_negative_pdf = os.path.join(
            OUTDIR, f"model2_omega_negative{i}.pdf"
        )
        plt.matshow(omega_negative / np.max(omega_negative))
        plt.savefig(omega_negative_pdf)

        phi_pdf = os.path.join(OUTDIR, f"model2_phi{i}.pdf")
        plt.matshow(phi / np.max(phi))
        plt.savefig(phi_pdf)

        scores = np.empty((iterations,))
        rand_scores = np.empty((iterations,))
        adjusted_rand_scores = np.empty((iterations,))
        for k in range(iterations):
            # generate a graph
            graph = PolarizationGraph.from_model2(
                n_nodes,
                n_threads,
                phi,
                omega_positive,
                omega_negative,
                theta,
                beta_a,
                beta_n,
            )
            # create the array encoding the communities from the number of
            # nodes.
            communities = []
            for j, n_group_nodes in enumerate(n_nodes):
                communities += [j] * n_group_nodes

            score, rand_score, adjusted_rand_score, duration = evaluate_graph(
                graph, alpha, n_communities, communities
            )

            scores[k] = score
            rand_scores[k] = rand_score
            adjusted_rand_scores[k] = adjusted_rand_score

        outfile = os.path.join(OUTDIR, f"model2_graph{i}.pdf")
        graph.draw(output=outfile, communities=communities)

        print_results(
            graph,
            omega_positive,
            omega_negative,
            scores,
            duration,
            rand_scores,
            adjusted_rand_scores,
        )

        i += 1


if __name__ == "__main__":
    if not os.path.exists(OUTDIR):
        dir1 = os.path.split(OUTDIR)[0]

        if not os.path.exists(dir1):
            os.mkdir(dir1)

        os.mkdir(OUTDIR)

    test_synthetic1(2)
    test_synthetic2(2)
