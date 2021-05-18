import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

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


def add_noise(signal: np.array, noise_sign: np.array, std_dev: float):
    # sample noise from truncated normal distribution
    noise = truncnorm.rvs(a=0, b=1, scale=std_dev, size=noise_sign.shape)

    noise = noise * noise_sign
    noised_signal = signal + noise

    return noised_signal


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
    n_members = 6

    # activation parameters
    theta = 0
    beta_a = 1
    beta_n = 1

    # controversy parameter
    alpha = 0.2

    n_communities = len(omega_positive_no_noise)
    n_nodes = [n_members] * n_communities

    # noise standard deviations:
    sigmas = np.arange(0, 1.1, 0.1)
    noise_sign = np.array(
        [
            [-1, 1, 1, 1],
            [1, -1, 1, 1],
            [1, 1, -1, 1],
            [1, 1, 1, -1],
        ]
    )

    sigmas_adj_rand_score = []
    sigmas_jaccard_score = []

    for sigma in sigmas:
        # generate and add noise to omega
        omega_positive = add_noise(omega_positive_no_noise, noise_sign, sigma)

        omega_positive_pdf = os.path.join(
            OUTDIR, f"model2_omega_positive_sigma_{sigma}.pdf"
        )
        plt.figure()
        plt.matshow(omega_positive / np.max(omega_positive))
        plt.savefig(omega_positive_pdf)
        plt.close()

        omega_negative = np.ones_like(omega_positive) - omega_positive
        omega_negative_pdf = os.path.join(
            OUTDIR, f"model2_omega_negative_sigma_{sigma}.pdf"
        )
        plt.figure()
        plt.matshow(omega_negative / np.max(omega_negative))
        plt.savefig(omega_negative_pdf)
        plt.close()

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

        outfile = os.path.join(OUTDIR, f"model2_graph_sigma_{sigma}.pdf")
        graph.draw(output=outfile, communities=communities)
        plotfilename = os.path.join(OUTDIR, f"model1_scores_sigma_{sigma}.pdf")

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

        sigmas_adj_rand_score.append(adjusted_rand_score)
        sigmas_jaccard_score.append(jaccard_score)

    sigmas_adj_rand_pdf = os.path.join(OUTDIR, f"model2_sigmas_adj_rand.pdf")
    plt.figure()
    plt.plot(sigmas, sigmas_adj_rand_score)
    plt.xlabel(r"$\sigma$")
    plt.ylabel("Adjusted RAND")
    plt.savefig(sigmas_adj_rand_pdf)
    plt.close()

    sigmas_jaccard_pdf = os.path.join(OUTDIR, f"model2_sigmas_jaccard.pdf")
    plt.figure()
    plt.plot(sigmas, sigmas_jaccard_score)
    plt.xlabel(r"$\sigma$")
    plt.ylabel("Jaccard")
    plt.savefig(sigmas_jaccard_pdf)
    plt.close()


if __name__ == "__main__":
    if not os.path.exists(OUTDIR):
        dir1 = os.path.split(OUTDIR)[0]

        if not os.path.exists(dir1):
            os.mkdir(dir1)

        os.mkdir(OUTDIR)

    outfile = open(os.path.join(OUTDIR, "results3.txt"), "w")

    test_synthetic(outfile)

    outfile.close()
