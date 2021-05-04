import os
import numpy as np
import matplotlib.pyplot as plt

from polarmine.graph import PolarizationGraph
from lib_synthetic import evaluate_graph, print_results

OUTDIR = os.path.join("out", "synthetic")


"""
The following are generic tests run to get an overview about how the model and
measure behave in some example conditions
"""


def test_synthetic(results_outfile, iterations: int = 1):

    n_nodes_list = []
    n_threads_list = []
    omega_positive_list = []
    omega_negative_list = []
    phi_list = []
    theta_list = []
    beta_a_list = []
    beta_n_list = []

    # -----------------------------------
    # GRAPH 1
    # -----------------------------------

    n_nodes = [20, 20, 20, 20]
    omega_positive = np.array(
        [
            [0.9, 0.01, 0.01, 0.01],
            [0.01, 0.9, 0.01, 0.01],
            [0.01, 0.01, 0.9, 0.01],
            [0.01, 0.01, 0.01, 0.9],
        ]
    )
    omega_negative = np.ones_like(omega_positive) - omega_positive
    phi = np.array(
        [
            [0.7, 0.2, 0.2, 0.2],
            [0.2, 0.7, 0.2, 0.2],
            [0.2, 0.2, 0.7, 0.2],
            [0.2, 0.2, 0.2, 0.7],
        ]
    )
    n_nodes_list.append(n_nodes)
    n_threads_list.append(18)
    omega_positive_list.append(np.array(omega_positive) / 8)
    omega_negative_list.append(np.array(omega_negative) / 8)
    phi_list.append(np.array(phi))
    theta_list.append(0.1)
    beta_a_list.append(1 / 8)
    beta_n_list.append(1 / 3)

    # -----------------------------------
    # GRAPH 2
    # -----------------------------------

    n_nodes = [20, 20, 20, 20]
    omega_positive = np.array(
        [
            [0.8, 0.03, 0.03, 0.03],
            [0.03, 0.8, 0.03, 0.03],
            [0.03, 0.03, 0.8, 0.03],
            [0.03, 0.03, 0.03, 0.8],
        ]
    )
    omega_negative = np.ones_like(omega_positive) - omega_positive
    phi = np.array(
        [
            [0.7, 0.2, 0.2, 0.2],
            [0.2, 0.7, 0.2, 0.2],
            [0.2, 0.2, 0.7, 0.2],
            [0.2, 0.2, 0.2, 0.7],
        ]
    )

    n_nodes_list.append(n_nodes)
    n_threads_list.append(8)
    omega_positive_list.append(np.array(omega_positive) / 4)
    omega_negative_list.append(np.array(omega_negative) / 8)
    phi_list.append(np.array(phi))
    theta_list.append(0.1)
    beta_a_list.append(1 / 8)
    beta_n_list.append(1 / 3)

    # -----------------------------------
    # GRAPH 3
    # -----------------------------------

    n_nodes = [20, 20, 20, 20]
    omega_positive = np.array(
        [
            [0.7, 0.1, 0.1, 0.1],
            [0.1, 0.7, 0.1, 0.1],
            [0.1, 0.1, 0.7, 0.1],
            [0.1, 0.1, 0.1, 0.7],
        ]
    )
    omega_negative = np.ones_like(omega_positive) - omega_positive
    phi = np.array(
        [
            [0.6, 0.4, 0.4, 0.4],
            [0.4, 0.6, 0.4, 0.4],
            [0.4, 0.4, 0.6, 0.4],
            [0.4, 0.4, 0.4, 0.6],
        ]
    )

    n_nodes_list.append(n_nodes)
    n_threads_list.append(8)
    omega_positive_list.append(np.array(omega_positive) / 32)
    omega_negative_list.append(np.array(omega_negative) / 64)
    phi_list.append(np.array(phi) / 8)
    theta_list.append(0.1)
    beta_a_list.append(1 / 1)
    beta_n_list.append(1 / 3)

    # -----------------------------------
    # GRAPH 4
    # -----------------------------------

    n_nodes = [20, 20, 20, 20]
    omega_positive = np.array(
        [
            [0.5, 0.4, 0.4, 0.4],
            [0.4, 0.5, 0.4, 0.4],
            [0.4, 0.4, 0.5, 0.4],
            [0.4, 0.4, 0.4, 0.5],
        ]
    )
    omega_negative = np.ones_like(omega_positive) - omega_positive
    phi = np.array(
        [
            [0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5],
        ]
    )

    n_nodes_list.append(n_nodes)
    n_threads_list.append(8)
    omega_positive_list.append(np.array(omega_positive) / 8)
    omega_negative_list.append(np.array(omega_negative) / 8)
    phi_list.append(np.array(phi))
    theta_list.append(0.1)
    beta_a_list.append(1 / 8)
    beta_n_list.append(1 / 3)

    i = 0
    alpha = 0.2

    scores = np.empty((iterations,))
    rand_scores = np.empty((iterations,))
    adjusted_rand_scores = np.empty((iterations,))
    jaccard_scores = np.empty((iterations,))

    for (
        n_nodes,
        n_threads,
        omega_positive,
        omega_negative,
        phi,
        theta,
        beta_a,
        beta_n,
    ) in zip(
        n_nodes_list,
        n_threads_list,
        omega_positive_list,
        omega_negative_list,
        phi_list,
        theta_list,
        beta_a_list,
        beta_n_list,
    ):
        for n_members in range(10, 100, 30):
            n_communities = len(omega_negative)
            n_nodes = [n_members] * n_communities

            omega_positive_pdf = os.path.join(
                OUTDIR, f"model2_omega_positive{i}_{n_members}.pdf"
            )
            plt.matshow(omega_positive / np.max(omega_positive))
            plt.savefig(omega_positive_pdf)
            plt.close()

            omega_negative_pdf = os.path.join(
                OUTDIR, f"model2_omega_negative{i}_{n_members}.pdf"
            )
            plt.matshow(omega_negative / np.max(omega_negative))
            plt.savefig(omega_negative_pdf)
            plt.close()

            phi_pdf = os.path.join(OUTDIR, f"model2_phi{i}_{n_members}.pdf")
            plt.matshow(phi / np.max(phi))
            plt.savefig(phi_pdf)
            plt.close()

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

                (
                    score,
                    rand_score,
                    adjusted_rand_score,
                    jaccard_score,
                    iterations_score,
                    duration,
                ) = evaluate_graph(graph, alpha, n_communities, communities)

                scores[k] = score
                rand_scores[k] = rand_score
                adjusted_rand_scores[k] = adjusted_rand_score
                jaccard_scores[k] = jaccard_score

            outfile = os.path.join(OUTDIR, f"model2_graph{i}_{n_members}.pdf")
            graph.draw(output=outfile, communities=communities)
            plotfilename = os.path.join(
                OUTDIR, f"model1_scores{i}_{n_members}.pdf"
            )

            print_results(
                graph,
                omega_positive,
                omega_negative,
                scores,
                duration,
                rand_scores,
                adjusted_rand_scores,
                jaccard_scores,
                iterations_score,
                results_outfile,
                plotfilename,
            )

            i += 1


if __name__ == "__main__":
    if not os.path.exists(OUTDIR):
        dir1 = os.path.split(OUTDIR)[0]

        if not os.path.exists(dir1):
            os.mkdir(dir1)

        os.mkdir(OUTDIR)

    outfile = open(os.path.join(OUTDIR, "results2.txt"), "w")

    test_synthetic(outfile, 2)

    outfile.close()
