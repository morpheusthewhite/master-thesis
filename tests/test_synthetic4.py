import os
import numpy as np
import matplotlib.pyplot as plt

from polarmine.graph import InteractionGraph
from lib_synthetic import (
    evaluate_graph,
    print_results,
)

OUTDIR = os.path.join("out", "synthetic")


def test_synthetic(results_outfile, n_iterations: int = 5):

    omega_positive_no_noise = np.array(
        [
            [0.7, 0, 0, 0],
            [0, 0.7, 0, 0],
            [0, 0, 0.7, 0],
            [0, 0, 0, 0.7],
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

    n_members = 8
    n_communities = len(omega_positive_no_noise)
    n_nodes = [n_members] * n_communities
    #  n_nodes = [7, 6, 5, 4]

    # noise standard deviations:
    noise_values = np.arange(0, 0.6, 0.1)

    # probability of having an edge between different communities
    edge_p = 0.05
    noise_multiplier = np.array(
        [
            [-0.7, edge_p, edge_p, edge_p],
            [edge_p, -0.7, edge_p, edge_p],
            [edge_p, edge_p, -0.7, edge_p],
            [edge_p, edge_p, edge_p, -0.7],
        ]
    )
    # probabilities of having an edge (positive or negative)
    edge_probabities = np.array(
        [
            [0.7, edge_p, edge_p, edge_p],
            [edge_p, 0.7, edge_p, edge_p],
            [edge_p, edge_p, 0.7, edge_p],
            [edge_p, edge_p, edge_p, 0.7],
        ]
    )

    noise_adj_rand_score = []
    noise_adj_rand_score_dev = []
    noise_jaccard_score = []
    noise_jaccard_score_dev = []

    scores = np.empty((n_iterations,))
    rand_scores = np.empty((n_iterations,))
    adjusted_rand_scores = np.empty((n_iterations,))
    jaccard_scores = np.empty((n_iterations,))

    for noise_value in noise_values:
        # generate and add noise to omega
        noise = noise_value * noise_multiplier
        omega_positive = omega_positive_no_noise + noise

        omega_positive_pdf = os.path.join(
            OUTDIR, f"model2_omega_positive_noise_{noise_value}.pdf"
        )
        plt.matshow(omega_positive / np.max(omega_positive), fignum=0)
        plt.savefig(omega_positive_pdf)
        plt.clf()

        omega_negative = edge_probabities - omega_positive
        omega_negative_pdf = os.path.join(
            OUTDIR, f"model2_omega_negative_noise_{noise_value}.pdf"
        )
        plt.matshow(omega_negative / np.max(omega_negative), fignum=0)
        plt.savefig(omega_negative_pdf)
        plt.clf()

        # create the array encoding the communities from the number of
        # nodes.
        communities = []
        for j, n_group_nodes in enumerate(n_nodes):
            communities += [j] * n_group_nodes

        for k in range(n_iterations):
            # generate a graph
            graph = InteractionGraph.from_model2(
                n_nodes,
                n_threads,
                phi,
                omega_positive,
                omega_negative,
                theta,
                beta_a,
                beta_n,
            )

            (
                score,
                rand_score,
                adjusted_rand_score,
                jaccard_score,
                iterations_score,
                purities,
                duration,
            ) = evaluate_graph(
                graph,
                alpha,
                n_communities,
                communities,
            )

            scores[k] = score
            rand_scores[k] = rand_score
            adjusted_rand_scores[k] = adjusted_rand_score
            jaccard_scores[k] = jaccard_score

        outfile_graph = os.path.join(
            OUTDIR, f"model2_graph_noise_{noise_value}.pdf"
        )
        graph.draw(output=outfile_graph, communities=communities)
        plotfilename = os.path.join(
            OUTDIR, f"model1_scores_noise_{noise_value}.pdf"
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
            purities,
            results_outfile,
            plotfilename,
        )
        results_outfile.flush()

        noise_adj_rand_score.append(np.average(adjusted_rand_scores))
        noise_adj_rand_score_dev.append(np.std(adjusted_rand_scores))
        noise_jaccard_score.append(np.average(jaccard_score))
        noise_jaccard_score_dev.append(np.std(adjusted_rand_scores))

    noise_adj_rand_pdf = os.path.join(OUTDIR, "model2_noise_adj_rand.pdf")
    plt.errorbar(noise_values, noise_adj_rand_score, noise_adj_rand_score_dev)
    plt.xlabel("noise")
    plt.ylabel("Adjusted RAND")
    plt.savefig(noise_adj_rand_pdf)
    plt.clf()

    noise_jaccard_pdf = os.path.join(OUTDIR, "model2_noise_jaccard.pdf")
    plt.errorbar(noise_values, noise_jaccard_score, noise_jaccard_score_dev)
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
