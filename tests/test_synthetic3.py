import os
import numpy as np
import matplotlib.pyplot as plt

from polarmine.graph import PolarizationGraph
from lib_synthetic import (
    evaluate_graph,
    print_results,
    CLUSTERING_EXACT,
    CLUSTERING_02_BFF,
    CLUSTERING_APPROXIMATION,
    CLUSTERING_NC_SUBGRAPH,
)

OUTDIR = os.path.join("out", "synthetic")


def test_synthetic(results_outfile):

    omega_positive_no_noise = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    phi = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    # graph dimension parameters
    n_threads = 2

    # activation parameters
    theta = 0
    beta_a = 1
    beta_n = 1

    # controversy parameter
    alpha = 0.2

    #  n_members = 6
    n_communities = len(omega_positive_no_noise)
    #  n_nodes = [n_members] * n_communities
    n_nodes = [7, 6, 5, 4]

    # noise standard deviations:
    noise_values = np.arange(0, 1.1, 0.1)
    noise_sign = np.array(
        [
            [-1, 1, 1, 1],
            [1, -1, 1, 1],
            [1, 1, -1, 1],
            [1, 1, 1, -1],
        ]
    )

    noise_adj_rand_score = []
    noise_jaccard_score = []

    for noise_value in noise_values:
        # generate and add noise to omega
        noise = noise_value * noise_sign
        omega_positive = omega_positive_no_noise + noise

        omega_positive_pdf = os.path.join(
            OUTDIR, f"model2_omega_positive_noise_{noise_value}.pdf"
        )
        plt.matshow(omega_positive / np.max(omega_positive), fignum=0)
        plt.savefig(omega_positive_pdf)
        plt.clf()

        omega_negative = np.ones_like(omega_positive) - omega_positive
        omega_negative_pdf = os.path.join(
            OUTDIR, f"model2_omega_negative_noise_{noise_value}.pdf"
        )
        plt.matshow(omega_negative / np.max(omega_negative), fignum=0)
        plt.savefig(omega_negative_pdf)
        plt.clf()

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
        ) = evaluate_graph(
            graph, alpha, n_communities, communities, CLUSTERING_02_BFF
        )

        outfile_graph = os.path.join(
            OUTDIR, f"model2_graph_noise_{noise_value}.pdf"
        )
        graph.draw(output=outfile_graph, communities=communities)
        plotfilename = os.path.join(OUTDIR, f"model1_scores_noise_{noise}.pdf")

        print_results(
            graph,
            omega_positive,
            omega_negative,
            score,
            duration,
            rand_score,
            adjusted_rand_score,
            jaccard_score,
            iterations_score,
            results_outfile,
            plotfilename,
        )
        results_outfile.flush()

        noise_adj_rand_score.append(adjusted_rand_score)
        noise_jaccard_score.append(jaccard_score)

    noise_adj_rand_pdf = os.path.join(OUTDIR, "model2_noise_adj_rand.pdf")
    plt.plot(noise_values, noise_adj_rand_score)
    plt.xlabel("noise")
    plt.ylabel("Adjusted RAND")
    plt.savefig(noise_adj_rand_pdf)
    plt.clf()

    noise_jaccard_pdf = os.path.join(OUTDIR, "model2_noise_jaccard.pdf")
    plt.plot(noise_values, noise_jaccard_score)
    plt.xlabel("noise")
    plt.ylabel("Jaccard")
    plt.savefig(noise_jaccard_pdf)
    plt.clf()


if __name__ == "__main__":
    if not os.path.exists(OUTDIR):
        dir1 = os.path.split(OUTDIR)[0]

        if not os.path.exists(dir1):
            os.mkdir(dir1)

        os.mkdir(OUTDIR)

    outfile = open(os.path.join(OUTDIR, "results3.txt"), "w")

    test_synthetic(outfile)

    outfile.close()
