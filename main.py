import argparse
import itertools
import matplotlib.pyplot as plt
import sys
import os

from polarmine.graph import PolarizationGraph
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
    default=None,
    action="store_true",
    dest="stats",
    help="show or save common graph statistics",
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


def print_stats(graph, save_path):
    graph.remove_self_loops()

    if save_path is None:
        stats_txt_file = sys.stdout
    else:
        stats_txt = os.path.join(save_path, "stats.txt")
        stats_txt_file = open(stats_txt, "w")

    print(
        f"The graph has {graph.num_vertices()} vertices and {graph.num_edges()} edges",
        file=stats_txt_file,
    )
    print(
        f"Fraction of nodes in k-core: {graph.kcore_size()}",
        file=stats_txt_file,
    )
    print(
        f"Fraction of negative edges: {graph.negative_edges_fraction()}",
        file=stats_txt_file,
    )

    global_clustering, global_clustering_stddev = graph.global_clustering()
    print(
        f"Clustering coefficient: {global_clustering} with standard deviation {global_clustering_stddev}",
        file=stats_txt_file,
    )

    print(
        f"Average shortest path length: {graph.average_shortest_path_length()}",
        file=stats_txt_file,
    )
    print(
        f"Median shortest path length: {graph.median_shortest_path_length()}",
        file=stats_txt_file,
    )
    print(f"Average degree: {graph.average_degree()}", file=stats_txt_file)
    print(
        f"Unique average degree: {graph.average_degree(unique=True)}",
        file=stats_txt_file,
    )

    print(
        f"Frustration index: {graph.social_balance()}",
        file=stats_txt_file,
    )

    # show degree histogram
    degrees = graph.degree_values()
    plt.figure()
    plt.title("Degree histogram")
    plt.hist(degrees)
    plt.xlabel("Degree")
    plt.ylabel("Number of nodes")

    if save_path is not None:
        degree_hist_pdf = os.path.join(save_path, "degree-hist.pdf")
        plt.savefig(degree_hist_pdf)
    else:
        plt.show()
        plt.close()

    # show negative edge fraction histogram for threads
    thread_fractions_dict = graph.negative_edges_fraction_thread_dict()
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
    print_negative_fraction_top_k(
        thread_fractions_dict, stats_txt_file, key="thread"
    )

    # show negative edge fraction histogram for threads
    content_fractions_dict = graph.negative_edges_fraction_content_dict()
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
    print_negative_fraction_top_k(
        content_fractions_dict, stats_txt_file, key="content"
    )

    # show degree distribution
    cum_probabilities, bins = graph.degree_distribution()
    plt.figure()
    plt.title("Degree cumulative distribution")
    plt.plot(bins, cum_probabilities)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Degree")
    plt.ylabel("Fraction of nodes with larger degree")

    if save_path is not None:
        degree_dist_pdf = os.path.join(save_path, "degree-dist.pdf")
        plt.savefig(degree_dist_pdf)
    else:
        plt.show()
        plt.close()

    # show user fidelity histogram
    fidelities = graph.fidelity_values()
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

    # show number of interactions histogram
    edge_sum_n_interactions = graph.edge_sum_n_interactions_values()
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

    # show content standard dev
    content_std_dev_dict = graph.content_std_dev_dict()
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

    if save_path is not None:
        stats_txt_file.close()


def main():

    # either load the graph or mine it
    if args.load is not None:
        graph = PolarizationGraph.from_file(args.load)
    else:
        # mine data and store it
        contents = iter([])

        if args.r or args.r_kw is not None or args.r_pg is not None:
            reddit_collector = RedditCollector()
            reddit_iter = reddit_collector.collect(
                args.rn, args.r_kw, args.r_pg, limit=args.rl, cross=args.rc
            )

            contents = itertools.chain(contents, reddit_iter)

        if args.t_kw is not None or args.t_pg is not None:
            twitter_collector = TwitterCollector()
            twitter_iter = twitter_collector.collect(
                args.tn, args.t_kw, args.t_pg, limit=args.tl, cross=args.tc
            )

            contents = itertools.chain(contents, twitter_iter)

        graph = PolarizationGraph(contents)

        if args.dump is not None:
            graph.dump(args.dump)

    if args.k > 0:
        graph.select_kcore(args.k)

    if args.save_path is not None and not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    if args.stats:
        print_stats(graph, args.save_path)

    if not args.graph_draw_no and args.save_path is not None:
        graph_output_path = os.path.join(args.save_path, "graph.pdf")
        graph.draw(output=graph_output_path)
    elif not args.graph_draw_no and not args.stats:
        # avoid plotting is stats is true and plots are not saved since
        # it raises a segmentation error
        graph.draw()


if __name__ == "__main__":
    main()
