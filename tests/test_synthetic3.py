import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

from polarmine.graph import PolarizationGraph
from lib_synthetic import evaluate_graph, print_results

OUTDIR = os.path.join("out", "synthetic")


def add_noise(signal: np.array, noise_sign: np.array, std_dev: float):
    # sample noise from truncated normal distribution
    noise = truncnorm.rvs(a=0, b=1, scale=std_dev, size=noise_sign.shape)

    noise = noise * noise_sign
    noised_signal = signal + noise

    return noised_signal


def test_synthetic(results_outfile, iterations: int = 1):

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
    n_threads = 3
    n_members = 4

    # activation parameters
    theta = 0
    beta_a = 1
    beta_n = 1

    scores = np.empty((iterations,))
    rand_scores = np.empty((iterations,))
    adjusted_rand_scores = np.empty((iterations,))
    jaccard_scores = np.empty((iterations,))

    i = 0
    alpha = 0.2

    n_communities = len(omega_positive_no_noise)
    n_nodes = [n_members] * n_communities

    # noise standard deviations:
    sigmas = [0, 0.1, 0.5, 1]
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
            OUTDIR, f"model2_omega_positive{i}.pdf"
        )
        plt.figure()
        plt.matshow(omega_positive / np.max(omega_positive))
        plt.savefig(omega_positive_pdf)
        plt.close()

        omega_negative = np.ones_like(omega_positive) - omega_positive
        omega_negative_pdf = os.path.join(
            OUTDIR, f"model2_omega_negative{i}.pdf"
        )
        plt.figure()
        plt.matshow(omega_negative / np.max(omega_negative))
        plt.savefig(omega_negative_pdf)
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
            ) = evaluate_graph(graph, alpha, n_communities, communities, False)

            scores[k] = score
            rand_scores[k] = rand_score
            adjusted_rand_scores[k] = adjusted_rand_score
            jaccard_scores[k] = jaccard_score

        outfile = os.path.join(OUTDIR, f"model2_graph{i}.pdf")
        graph.draw(output=outfile, communities=communities)
        plotfilename = os.path.join(OUTDIR, f"model1_scores{i}.pdf")

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

        sigmas_adj_rand_score.append(np.average(adjusted_rand_scores))
        sigmas_jaccard_score.append(np.average(jaccard_scores))

        i += 1

    sigmas_adj_rand_pdf = os.path.join(OUTDIR, f"model2_sigmas_adj_rand.pdf")
    plt.figure()
    plt.plot(sigmas, sigmas_adj_rand_score)
    plt.savefig(sigmas_adj_rand_pdf)
    plt.close()

    sigmas_jaccard_pdf = os.path.join(OUTDIR, f"model2_sigmas_jaccard.pdf")
    plt.figure()
    plt.plot(sigmas, sigmas_jaccard_score)
    plt.savefig(sigmas_jaccard_pdf)
    plt.close()


if __name__ == "__main__":
    if not os.path.exists(OUTDIR):
        dir1 = os.path.split(OUTDIR)[0]

        if not os.path.exists(dir1):
            os.mkdir(dir1)

        os.mkdir(OUTDIR)

    outfile = open(os.path.join(OUTDIR, "results3.txt"), "w")

    test_synthetic(outfile, 2)

    outfile.close()
