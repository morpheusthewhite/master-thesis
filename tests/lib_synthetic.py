from typing import Optional
import time
import os
import numpy as np
import matplotlib.pyplot as plt

from polarmine.ecp.ecp_solver import ECPSolver
from polarmine.ecp import ECPRoundingSolver
from polarmine.validation import clustering_accuracy
from polarmine.graph import (
    InteractionGraph,
)

OUTDIR = os.path.join("out", "synthetic")


def evaluate_graph(
    graph: InteractionGraph,
    alpha: float,
    n_communities: int,
    communities: list[int],
    solver: Optional[ECPSolver] = None,
):
    if solver is None:
        solver = ECPRoundingSolver()

    start = time.time()
    score, _, _, _ = solver.solve(graph, alpha)

    (
        adjusted_rand_score,
        rand_score,
        jaccard_score,
        iterations_score,
        purity_score,
        _,
    ) = clustering_accuracy(graph, communities, n_communities, alpha, solver)
    end = time.time()

    return (
        score,
        rand_score,
        adjusted_rand_score,
        jaccard_score,
        iterations_score,
        purity_score,
        end - start,
    )


def print_results(
    graph: InteractionGraph,
    omega_positive: np.array,
    omega_negative: np.array,
    scores: np.array,
    duration: int,
    rand_scores: np.array,
    adjusted_rand_scores: np.array,
    jaccard_scores: np.array,
    iterations_score: list[float],
    outfile,
    plotfilename,
):
    # Will save to file only one of the graphs
    print("-" * 30, file=outfile)
    print(
        f"Vertices: {graph.num_vertices()}; Edges: {graph.num_edges()}",
        file=outfile,
    )
    print(f"Omega positive: {omega_positive}", file=outfile)
    print(f"Omega negative: {omega_negative}", file=outfile)
    print(
        f"Fraction of negative edges: {graph.negative_edges_fraction()}",
        file=outfile,
    )
    print(f"Score MIP: {np.average(scores)}", file=outfile)
    print(f"Time: {duration}", file=outfile)
    print(f"Clustering Rand score: {np.average(rand_scores)}", file=outfile)
    print(
        f"Clustering Adjusted Rand score: {np.average(adjusted_rand_scores)}",
        file=outfile,
    )
    print(
        f"Jaccard score: {np.average(jaccard_scores)}",
        file=outfile,
    )

    # plot scores along iterations
    plt.plot(iterations_score)
    plt.savefig(plotfilename)
    plt.clf()

    print("-" * 30, file=outfile)

    return
