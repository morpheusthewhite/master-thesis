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


def print_top_k(
    top_k_dictionary,
    outfile,
    key: str = "thread",
    value: str = "negative edge fraction",
    k=3,
    reverse: bool = False,
):
    top_k_sorted_dictionary = {
        k: v
        for k, v in sorted(
            top_k_dictionary.items(),
            key=lambda item: item[1],
            reverse=reverse,
        )
    }

    # keys are either contents or threads
    keys = list(top_k_sorted_dictionary.keys())
    k = min(k, len(keys))

    if reverse:
        ranking = "highest"
    else:
        ranking = "lowest"

    print(
        f"{key.capitalize()} {ranking} {value} top {k}:",
        file=outfile,
    )
    for i in range(k):
        key_ith = keys[i]

        print(
            f"\t{key_ith} with {top_k_sorted_dictionary[key_ith]}",
            file=outfile,
        )
