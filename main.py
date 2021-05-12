import argparse
import itertools
import matplotlib.pyplot as plt
import sys
import os
import pickle
import time
import numpy as np
from typing import Optional

from polarmine.graph import PolarizationGraph
from polarmine.utils import plot_degree_distribution, print_top_k
from polarmine.collectors.reddit_collector import RedditCollector
from polarmine.collectors.twitter_collector import TwitterCollector


parser = argparse.ArgumentParser(description="Polarmine")

# graph
save_load_group = parser.add_mutually_exclusive_group()
save_load_group.add_argument(
    "--dump",
    "-d",
    type=str,
    default=None,
    metavar="filename",
    help="dump the mined graph at the given path",
)
save_load_group.add_argument(
    "--load",
    "-l",
    type=str,
    default=None,
    metavar="filename",
    help="load the mined graph at the given path",
)
parser.add_argument(
    "--label",
    default=False,
    action="store_true",
    dest="label",
    help="label selected nodes and save the graph",
)
parser.add_argument(
    "--draw-no",
    "-dn",
    default=None,
    action="store_true",
    dest="graph_draw_no",
    help="if provided does not show or save the graph",
)
parser.add_argument(
    "-s",
    "--stats",
    default=False,
    action="store_true",
    dest="stats",
    help="show or save common graph statistics",
)
parser.add_argument(
    "-sg",
    "--score-greedy",
    default=False,
    action="store_true",
    dest="score_greedy",
    help="show or save echo chamber greedy scores",
)
parser.add_argument(
    "-se",
    "--score-exact",
    default=False,
    action="store_true",
    dest="score_mip",
    help="show or save exact MIP echo chamber scores",
)
parser.add_argument(
    "-sa",
    "--score-appr",
    default=False,
    action="store_true",
    dest="score_appr",
    help="show or save appromiximation echo chamber score (from MIP relaxation results)",
)
parser.add_argument(
    "-sb",
    "--score-bff",
    default=False,
    action="store_true",
    dest="score_bff",
    help="show or save O2-BFF score (DCS-AM)",
)
parser.add_argument(
    "-scl",
    "--score-clustering",
    default=False,
    action="store_true",
    dest="score_clustering",
    help="cluster nodes through repeated ECP solutions and save or show accuracy",
)
parser.add_argument(
    "-a",
    "--alpha",
    default=0.4,
    type=float,
    dest="alpha",
    help="maximum fraction of negative edges of non controversial content. If -1 it is chosen as the median of the fraction of negative edges of the contents. If -2 the scores are computed for many different values",
)
parser.add_argument(
    "-sp",
    "--save-path",
    type=str,
    default=None,
    dest="save_path",
    metavar="path",
    help="save statistics (if --stats is passed) and graph drawing at the given path",
)
parser.add_argument(
    "-k",
    "--k-core",
    type=int,
    default=0,
    metavar="k",
    dest="k",
    help="k used to select k-core component for analysis and results",
)
# reddit
reddit_group = parser.add_mutually_exclusive_group()
reddit_group.add_argument(
    "-r",
    "--reddit",
    default=None,
    action="store_true",
    dest="r",
    help="mine data from reddit without filters",
)
reddit_group.add_argument(
    "-rkw",
    "--reddit-kw",
    type=str,
    default=None,
    metavar="reddit_keywords",
    dest="r_kw",
    help="keyword used for filtering reddit posts",
)
reddit_group.add_argument(
    "-rp",
    "--reddit-page",
    type=str,
    default=None,
    metavar="reddit_page",
    dest="r_pg",
    help="subreddit(s) from which posts are mined",
)
parser.add_argument(
    "-rl",
    "--reddit-limit",
    type=int,
    default=10,
    metavar="reddit_limit",
    dest="rl",
    help="max number of mined comments when looking for replies",
)
parser.add_argument(
    "-rn",
    "--reddit-n",
    type=int,
    default=1,
    metavar="reddit_n",
    dest="rn",
    help="number of mined posts",
)
parser.add_argument(
    "-rc",
    "--reddit-cross",
    action="store_true",
    default=False,
    dest="rc",
    help="if provided graph will also include crossposts",
)
# twitter
twitter_group = parser.add_mutually_exclusive_group()
twitter_group.add_argument(
    "-tkw",
    "--twitter-kw",
    type=str,
    default=None,
    metavar="twitter_keywords",
    dest="t_kw",
    help="keyword used for filtering tweets",
)
twitter_group.add_argument(
    "-tp",
    "--twitter-page",
    type=str,
    default=None,
    metavar="twitter_page",
    dest="t_pg",
    help="Twitter username from which posts are mined",
)
parser.add_argument(
    "-tl",
    "--twitter-limit",
    type=int,
    default=10,
    metavar="twitter_limit",
    dest="tl",
    help="max number of mined tweets when looking for replies",
)
parser.add_argument(
    "-tn",
    "--twitter-n",
    type=int,
    default=1,
    metavar="twitter_n",
    dest="tn",
    help="number of mined tweets",
)
parser.add_argument(
    "-tc",
    "--twitter-cross",
    action="store_true",
    default=False,
    dest="tc",
    help="if provided graph will also include reply quotes",
)

args = parser.parse_args()


def score_clustering(
    graph: PolarizationGraph, alpha: float, save_path: str = None
):
    if save_path is None:
        clustering_txt_file = sys.stdout
    else:
        clustering_txt = os.path.join(save_path, "clustering.txt")
        clustering_txt_file = open(clustering_txt, "w")

    if alpha == -1:
        alphas = [graph.alpha_median()]
        print(
            f"Median alpha of the graph: {alphas[0]}",
            file=clustering_txt_file,
        )
    elif alpha == -2:
        alpha_median = graph.alpha_median()
        print(
            f"Median alpha of the graph: {alpha_median}",
            file=clustering_txt_file,
        )

        alphas = [alpha_median] + list(np.arange(0.1, 1, 0.1))
    else:
        alphas = [alpha]

    (
        adj_rand_score,
        rand_score,
        jaccard,
        jaccard_iterations,
        precision_iterations,
        iteration_vertices,
    ) = graph.clustering_accuracy(graph.labels.a, 2, alpha)

    print(f"Adjusted RAND score: {adj_rand_score}", file=clustering_txt_file)
    print(f"RAND score: {rand_score}", file=clustering_txt_file)
    print(f"Jaccard score: {jaccard}", file=clustering_txt_file)

    plt.figure()
    plt.title("Jaccard score over iterations")
    plt.plot(jaccard_iterations)
    plt.xlabel("Iteration number")
    plt.ylabel("Jaccard score")

    if save_path is not None:
        jaccard_iterations_pdf = os.path.join(
            save_path, "jaccard_iterations.pdf"
        )
        plt.savefig(jaccard_iterations_pdf)
    else:
        plt.show()
        plt.close()

    plt.figure()
    plt.title("Precision score over iterations")
    plt.plot(precision_iterations)
    plt.xlabel("Iteration number")
    plt.ylabel("Precision score")

    if save_path is not None:
        precision_iterations_pdf = os.path.join(
            save_path, "precision_iterations.pdf"
        )
        plt.savefig(precision_iterations_pdf)
    else:
        plt.show()
        plt.close()

    if save_path is not None:
        clustering_txt_file.close()
    return


def print_negative_fraction_top_k(
    negative_edges_fraction_dict, file_, key="thread", k=3
):
    negative_edges_fraction_dict_sorted = {
        k: v
        for k, v in sorted(
            negative_edges_fraction_dict.items(), key=lambda item: item[1]
        )
    }

    # keys are either contents or threads
    keys = list(negative_edges_fraction_dict_sorted.keys())
    k = min(k, len(keys))

    print(
        f"{key.capitalize()} lowest negative edge fraction top {k}:",
        file=file_,
    )
    for i in range(k):
        key_ith = keys[i]

        print(
            f"\t{key_ith} with {negative_edges_fraction_dict_sorted[key_ith]}",
            file=file_,
        )

    print(
        f"{key.capitalize()} highest negative edge fraction top {k}:",
        file=file_,
    )
    for i in range(1, k + 1):
        key_ith = keys[-i]

        print(
            f"\t{key_ith} with {negative_edges_fraction_dict_sorted[key_ith]}",
            file=file_,
        )


def print_scores(
    graph: PolarizationGraph,
    greedy: bool,
    mip: bool,
    appr: bool,
    bff: bool,
    save_path: Optional[str],
    alpha: float = 0.4,
):
    if save_path is None:
        scores_txt_file = sys.stdout
        times_txt_file = sys.stdout
    else:
        scores_txt = os.path.join(save_path, "scores.txt")
        scores_txt_file = open(scores_txt, "w")
        times_txt = os.path.join(save_path, "times.txt")
        times_txt_file = open(times_txt, "w")

    if alpha == -1:
        alphas = [graph.alpha_median()]
        print(
            f"Median alpha of the graph: {alphas[0]}",
            file=scores_txt_file,
        )
    elif alpha == -2:
        alpha_median = graph.alpha_median()
        print(
            f"Median alpha of the graph: {alpha_median}", file=scores_txt_file
        )

        alphas = [alpha_median] + list(np.arange(0.1, 1, 0.1))
    else:
        alphas = [alpha]

    results_score = {}

    for alpha in alphas:
        print(
            f"The graph contains {graph.num_vertices()} vertices, {graph.num_edges()} edges and {len(graph.controversial_contents(alpha))} controversial contents for alpha={alpha}",
            file=scores_txt_file,
        )

        if greedy:
            start = time.time()
            score, users_index, nc_threads = graph.score_components(alpha)
            results_score[f"components_{alpha}"] = (score, users_index)
            print(
                f"(Connected components) Echo chamber score: {score} on {len(users_index)} vertices with {nc_threads} non controversial threads",
                file=scores_txt_file,
            )
            end = time.time()
            print(
                f"(Connected components) Elapsed time: {end - start}",
                file=times_txt_file,
            )

            results_greedy_beta_pos = {}
            for beta in [i / 10 for i in range(6, 11, 1)]:
                start = time.time()
                score, users_index, nc_threads = graph.score_greedy_beta(
                    alpha, beta
                )
                results_greedy_beta_pos[beta] = (score, users_index)
                print(
                    f"(Greedy beta={beta}, pos. sampling) Echo chamber score: {score} on {len(users_index)} vertices with {nc_threads} non controversial threads",
                    file=scores_txt_file,
                )
                end = time.time()
                print(
                    f"(Greedy beta={beta}, pos. sampling) Elapsed time: {end - start}",
                    file=times_txt_file,
                )

            results_score[f"greedy_beta_pos_{alpha}"] = results_greedy_beta_pos

            results_greedy_beta_uni = {}
            for beta in [i / 10 for i in range(6, 11, 1)]:
                start = time.time()
                score, users_index, nc_threads = graph.score_greedy_beta(
                    alpha, beta, positiveness_samples=False
                )
                results_greedy_beta_uni[beta] = (score, users_index)
                print(
                    f"(Greedy beta={beta}, unif. sampling) Echo chamber score: {score} on {len(users_index)} vertices with {nc_threads} non controversial threads",
                    file=scores_txt_file,
                )
                end = time.time()
                print(
                    f"(Greedy beta={beta}, unif. sampling) Elapsed time: {end - start}",
                    file=times_txt_file,
                )

            results_score[f"greedy_beta_uni_{alpha}"] = results_greedy_beta_uni

            start = time.time()
            score, users_index, nc_threads = graph.score_greedy_peeling(alpha)
            results_score[f"greedy_peeling_{alpha}"] = (score, users_index)
            print(
                f"(Greedy peeling) Echo chamber score: {score} on {len(users_index)} vertices with {nc_threads} non controversial threads",
                file=scores_txt_file,
            )
            end = time.time()
            print(
                f"(Greedy peeling) Elapsed time: {end - start}",
                file=times_txt_file,
            )

        if mip:
            # start = time.time()
            # score, users_index, _, nc_threads = graph.score_mip(
            #     alpha, relaxation=True
            # )
            # results_score[f"mip_relaxation_{alpha"] = (score, users_index)
            # print(
            #     f"(MIP relaxation) Echo chamber score: {score} on {len(users_index)} vertices with {len(nc_threads)} non controversial threads",
            #     file=scores_txt_file,
            # )
            # end = time.time()
            # print(
            #     f"(MIP relaxation) Elapsed time: {end - start}",
            #     file=times_txt_file,
            # )

            start = time.time()
            score, users_index, edges, nc_threads = graph.score_mip(alpha)
            results_score[f"mip_{alpha}"] = (score, users_index, edges)
            print(
                f"(MIP) Echo chamber score: {score} on {len(users_index)} vertices with {len(nc_threads)} non controversial threads",
                file=scores_txt_file,
            )
            end = time.time()
            print(
                f"(MIP) Elapsed time: {end - start}",
                file=times_txt_file,
            )

            start = time.time()
            score, users_index, edges, nc_threads = graph.score_mip_densest(
                alpha
            )
            results_score[f"mip-densest_{alpha}"] = (score, users_index, edges)
            print(
                f"(MIP) Densest echo chamber score: {score} on {len(users_index)} vertices with {len(nc_threads)} non controversial threads",
                file=scores_txt_file,
            )
            end = time.time()
            print(
                f"(MIP) Elapsed time: {end - start}",
                file=times_txt_file,
            )

        if appr:
            start = time.time()
            score, users_index, nc_threads = graph.score_relaxation_algorithm(
                alpha
            )
            results_score[f"mip_rounding_algorithm_{alpha}"] = (
                score,
                users_index,
            )
            print(
                f"(MIP rounding algorithm) Echo chamber score: {score} on {len(users_index)} vertices with {nc_threads} non controversial threads",
                file=scores_txt_file,
            )
            end = time.time()
            print(
                f"(MIP rounding algorithm) Elapsed time: {end - start}",
                file=times_txt_file,
            )

            start = time.time()
            score, users_index = graph.score_densest_nc_subgraph(alpha)
            results_score[f"densest_nc_subgraph_simple_{alpha}"] = (
                score,
                users_index,
            )
            print(
                f"(Densest nc subgraph (unthreaded)) Echo chamber score: {score} on {len(users_index)} vertices",
                file=scores_txt_file,
            )
            end = time.time()
            print(
                f"(Densest nc subgraph (unthreaded)) Elapsed time: {end - start}",
                file=times_txt_file,
            )

            start = time.time()
            score, users_index = graph.score_densest_nc_subgraph(alpha, False)
            results_score[f"densest_nc_subgraph_{alpha}"] = (
                score,
                users_index,
            )
            print(
                f"(Densest nc subgraph (threaded)) Echo chamber score: {score} on {len(users_index)} vertices",
                file=scores_txt_file,
            )
            end = time.time()
            print(
                f"(Densest nc subgraph (threaded)) Elapsed time: {end - start}",
                file=times_txt_file,
            )

        if bff:
            start = time.time()
            n_contents = graph.num_contents(alpha)
            k = int(np.ceil(n_contents / 10))
            score, users_index = graph.o2_bff_dcs_am(alpha, k)
            results_score[f"bff_{alpha}"] = (
                score,
                users_index,
            )
            print(
                f"(O2-BFF(DCS-AM)) Echo chamber score: {score} on {len(users_index)} vertices for k={k}",
                file=scores_txt_file,
            )
            end = time.time()
            print(
                f"(O2-BFF(DCS-AM)) Elapsed time: {end - start}",
                file=times_txt_file,
            )

    if save_path is not None:
        scores_txt_file.close()

        pickle_filename = os.path.join(save_path, "scores.p")
        with open(pickle_filename, "wb") as pickle_file:
            pickle.dump(results_score, pickle_file)


def print_stats(graph: PolarizationGraph, save_path):
    # dictionary to be pickled
    results_stats = {}

    if save_path is None:
        stats_txt_file = sys.stdout
    else:
        stats_txt = os.path.join(save_path, "stats.txt")
        stats_txt_file = open(stats_txt, "w")

    results_stats["num_vertices"] = graph.num_vertices()
    results_stats["num_edges"] = graph.num_edges()
    results_stats["kcore_size"] = graph.kcore_size()
    results_stats["negative_edges_fraction"] = graph.negative_edges_fraction()

    print(
        f"The graph has {results_stats['num_vertices']} vertices and {results_stats['num_edges']} edges",
        file=stats_txt_file,
    )
    print(
        f"Fraction of nodes in k-core: {results_stats['kcore_size']}",
        file=stats_txt_file,
    )
    print(
        f"Fraction of negative edges: {results_stats['negative_edges_fraction']}",
        file=stats_txt_file,
    )

    global_clustering, global_clustering_stddev = graph.global_clustering()
    results_stats["global_clustering"] = global_clustering
    results_stats["global_clustering_stddev"] = global_clustering_stddev
    print(
        f"Clustering coefficient: {global_clustering} with standard deviation {global_clustering_stddev}",
        file=stats_txt_file,
    )

    results_stats[
        "average_shortest_path_length"
    ] = graph.average_shortest_path_length()
    results_stats[
        "median_shortest_path_length"
    ] = graph.median_shortest_path_length()
    results_stats["average_degree"] = graph.average_degree()
    results_stats["unique_average_degree"] = graph.average_degree(unique=True)

    print(
        f"Average shortest path length: {results_stats['average_shortest_path_length']}",
        file=stats_txt_file,
    )
    print(
        f"Median shortest path length: {results_stats['median_shortest_path_length']}",
        file=stats_txt_file,
    )
    print(
        f"Average degree: {results_stats['average_degree']}",
        file=stats_txt_file,
    )
    print(
        f"Unique average degree: {results_stats['unique_average_degree']}",
        file=stats_txt_file,
    )

    # show negative edge fraction histogram for threads
    thread_fractions_dict = graph.negative_edges_fraction_thread_dict()
    results_stats["thread_fractions_dict"] = thread_fractions_dict
    plt.figure()
    plt.title("Thread edge negativeness histogram")
    plt.hist(thread_fractions_dict.values())
    plt.xlabel("Negative edge fraction")
    plt.ylabel("Number of threads")

    if save_path is not None:
        neg_fraction_thread_hist_pdf = os.path.join(
            save_path, "neg-fraction-thread-hist.pdf"
        )
        plt.savefig(neg_fraction_thread_hist_pdf)
    else:
        plt.show()
        plt.close()

    # write the top-k (both ascending and descending) of the contents
    print_top_k(
        thread_fractions_dict,
        outfile=stats_txt_file,
        key="thread",
        value="negative edge fraction",
    )
    print_top_k(
        thread_fractions_dict,
        outfile=stats_txt_file,
        key="thread",
        value="negative edge fraction",
        reverse=True,
    )

    # show negative edge fraction histogram for threads
    content_fractions_dict = graph.negative_edges_fraction_content_dict()
    results_stats["content_fractions_dict"] = content_fractions_dict
    plt.figure()
    plt.title("Content edge negativeness histogram")
    plt.hist(content_fractions_dict.values())
    plt.xlabel("Negative edge fraction")
    plt.ylabel("Number of contents")

    if save_path is not None:
        neg_fraction_content_hist_pdf = os.path.join(
            save_path, "neg-fraction-content-hist.pdf"
        )
        plt.savefig(neg_fraction_content_hist_pdf)
    else:
        plt.show()
        plt.close()

    # write the top-k (both ascending and descending) of the contents
    print_top_k(
        content_fractions_dict,
        outfile=stats_txt_file,
        key="content",
        value="negative edge fraction",
    )
    print_top_k(
        content_fractions_dict,
        outfile=stats_txt_file,
        key="content",
        value="negative edge fraction",
        reverse=True,
    )

    plot_degree_distribution(graph, save_path, "total")
    plot_degree_distribution(graph, save_path, "in")
    plot_degree_distribution(graph, save_path, "out")

    # show user fidelity histogram
    fidelities = graph.fidelity_values()
    results_stats["fidelities"] = fidelities
    plt.figure()
    plt.title("User fidelity histogram")
    plt.hist(fidelities)
    plt.yscale("log")
    plt.xlabel("Number of different contents in which a user was involved")
    plt.ylabel("Number of users")

    if save_path is not None:
        fidelity_hist_pdf = os.path.join(save_path, "fidelity-hist.pdf")
        plt.savefig(fidelity_hist_pdf)
    else:
        plt.show()
        plt.close()

    # show number of interactions histogram
    n_interactions = graph.n_interaction_values()
    results_stats["n_interactions"] = n_interactions
    plt.figure()
    plt.title("Number of interactions histogram")
    plt.hist(n_interactions)
    plt.xlabel("Number of interactions")
    plt.ylabel("Number of contents")

    if save_path is not None:
        n_interactions_hist_pdf = os.path.join(
            save_path, "n-interactions-hist.pdf"
        )
        plt.savefig(n_interactions_hist_pdf)
    else:
        plt.show()
        plt.close()

    # show content standard dev
    content_std_dev_dict = graph.content_std_dev_dict()
    results_stats["content_std_dev_dict"] = content_std_dev_dict
    plt.figure()
    plt.title("Content standard deviation")
    plt.hist(list(content_std_dev_dict.values()))
    plt.xlabel("Standard deviation")
    plt.ylabel("Number of contents")

    if save_path is not None:
        content_std_dev_hist_pdf = os.path.join(
            save_path, "content-std-dev-hist.pdf"
        )
        plt.savefig(content_std_dev_hist_pdf)
    else:
        plt.show()
        plt.close()

    # show total edge sum over number of interactions
    edge_sum_n_interactions_dict = graph.edge_sum_n_interactions_dict()
    results_stats[
        "edge_sum_n_interactions_dict"
    ] = edge_sum_n_interactions_dict
    edge_sum_n_interactions = edge_sum_n_interactions_dict.values()
    x_n_interactions = [
        n_interactions for n_interactions, edge_sum in edge_sum_n_interactions
    ]
    y_edge_sum = [
        edge_sum for n_interactions, edge_sum in edge_sum_n_interactions
    ]
    plt.figure()
    plt.title("Content edge sum")
    plt.scatter(x_n_interactions, y_edge_sum)
    plt.xlabel("Number of interactions")
    plt.xlim(left=0)
    plt.ylabel("Total edge sum")

    # plot 2 lines, the bisectors of the 1st and 4th quadrants
    left, right = plt.xlim()
    plt.plot([0, right], [0, right], color="grey", linewidth=0.6)
    plt.plot([0, right], [0, -right], color="grey", linewidth=0.6)
    plt.plot([0, right], [0, 0], color="grey", linestyle=":", linewidth=0.5)

    if save_path is not None:
        edge_sum_n_interactions_pdf = os.path.join(
            save_path, "edge-sum-n-interactions.pdf"
        )
        plt.savefig(edge_sum_n_interactions_pdf)
    else:
        plt.show()
        plt.close()

    # merge dictionaries to plot std_dev over n_interactions and
    # std_dev over edge_sum
    edge_sum_n_interactions_std_dev_dict = {
        content: (
            edge_sum_n_interactions_dict[content][0],
            edge_sum_n_interactions_dict[content][1],
            std_dev,
        )
        for content, std_dev in content_std_dev_dict.items()
    }
    x_n_interactions = [
        n_interactions
        for n_interactions, edge_sum, std_dev in edge_sum_n_interactions_std_dev_dict.values()
    ]
    y_std_dev = [
        std_dev
        for n_interactions, edge_sum, std_dev in edge_sum_n_interactions_std_dev_dict.values()
    ]
    plt.figure()
    plt.title("Standard deviation over number of interactions")
    plt.scatter(x_n_interactions, y_std_dev)
    plt.xlabel("Number of interactions")
    plt.xlim(left=0)
    plt.ylabel("Standard deviation")

    if save_path is not None:
        std_dev_n_interactions_pdf = os.path.join(
            save_path, "std-dev-n-interactions.pdf"
        )
        plt.savefig(std_dev_n_interactions_pdf)
    else:
        plt.show()
        plt.close()

    x_edge_sum = [
        edge_sum
        for n_interactions, edge_sum, std_dev in edge_sum_n_interactions_std_dev_dict.values()
    ]
    plt.figure()
    plt.title("Standard deviation over edge sum")
    plt.scatter(x_edge_sum, y_std_dev)
    plt.xlabel("Edge sum")
    plt.ylabel("Standard deviation")

    if save_path is not None:
        std_dev_edge_sum_pdf = os.path.join(save_path, "std-dev-edge-sum.pdf")
        plt.savefig(std_dev_edge_sum_pdf)
    else:
        plt.show()
        plt.close()

    if save_path is not None:
        stats_txt_file.close()

        pickle_filename = os.path.join(save_path, "stats.p")
        with open(pickle_filename, "wb") as pickle_file:
            pickle.dump(results_stats, pickle_file)


def main():

    # either load the graph or mine it
    if args.load is not None:
        graph = PolarizationGraph.from_file(args.load)
    else:
        # mine data and store it
        contents = iter([])

        if args.t_kw is not None or args.t_pg is not None:
            twitter_collector = TwitterCollector()
            twitter_iter = twitter_collector.collect(
                args.tn, args.t_kw, args.t_pg, limit=args.tl, cross=args.tc
            )

            contents = itertools.chain(contents, twitter_iter)

        graph = PolarizationGraph(contents)

        if args.dump is not None:
            graph.dump(args.dump)

    if args.save_path is not None and not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    if args.k > 0:
        graph.select_kcore(args.k)

    if args.label:
        graph.label_nodes()

        if args.load is not None:
            graph.dump(args.load)
        elif args.dump is not None:
            graph.dump(args.dump)

    graph.remove_self_loops()

    if args.stats:
        print_stats(graph, args.save_path)

    if (
        args.score_greedy
        or args.score_mip
        or args.score_appr
        or args.score_bff
    ):
        print_scores(
            graph,
            args.score_greedy,
            args.score_mip,
            args.score_appr,
            args.score_bff,
            args.save_path,
            args.alpha,
        )

    if args.score_clustering:
        graph.deselect_unlabeled()
        graph.remove_isolated()
        score_clustering(graph, args.alpha, args.save_path)

    if not args.graph_draw_no and args.save_path is not None:
        graph_output_path = os.path.join(args.save_path, "graph.pdf")
        graph.draw(output=graph_output_path)
    elif not args.graph_draw_no and not args.stats:
        # avoid plotting is stats is true and plots are not saved since
        # it raises a segmentation error
        graph.draw()


if __name__ == "__main__":
    main()
