import os
import matplotlib.pyplot as plt

from polarmine.graph import PolarizationGraph


def plot_degree_distribution(
    graph: PolarizationGraph, save_path: str, degree: str = "total"
):
    # show degree distribution
    probabilities, bins = graph.degree_distribution(degree)
    plt.figure()
    plt.title(f"{degree.capitalize()} degree distribution")
    plt.plot(bins, probabilities)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(f"{degree.capitalize()} degree")
    plt.ylabel("Fraction of nodes")

    if save_path is not None:
        degree_dist_pdf = os.path.join(save_path, f"degree-{degree}-dist.pdf")
        plt.savefig(degree_dist_pdf)
    else:
        plt.show()
        plt.close()
