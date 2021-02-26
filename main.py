import argparse
import itertools
import matplotlib.pyplot as plt
import sys

from polarmine.graph import PolarizationGraph
from polarmine.collectors.reddit_collector import RedditCollector


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

draw_save_show_group = parser.add_mutually_exclusive_group()
draw_save_show_group.add_argument(
    "--draw-save",
    "-dv",
    type=str,
    default=None,
    dest="graph_draw_save",
    metavar="filename",
    help="save the graph drawing at a given path instead of showing it (extension must be .ps, .pdf, .svg, or .png)",
)
draw_save_show_group.add_argument(
    "--draw-no",
    "-dn",
    default=None,
    action="store_true",
    dest="graph_draw_no",
    help="if provided does not show the graph",
)
parser.add_argument(
    "-sh",
    "--stats-show",
    default=None,
    action="store_true",
    dest="stats_show",
    help="compute and show common graph statistics",
)
parser.add_argument(
    "-sv",
    "--stats-save",
    type=str,
    default=None,
    dest="stats_save",
    metavar="file-prefix",
    help="compute and save common graph statistics using the given prefix",
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


def compute_stats(graph, file_prefix):
    graph.remove_self_loops()

    if file_prefix is None:
        stats_txt_file = sys.stdout
    else:
        stats_txt = file_prefix + "-stats.txt"
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

    if file_prefix is not None:
        stats_txt_file.close()

    # show degree histogram
    # matplotlib is apparently segfaulting without a good reason
    counts, bins = graph.degree_histogram()
    plt.figure()
    plt.bar(bins, counts)

    if file_prefix is not None:
        hist_pdf = file_prefix + "-hist.pdf"
        plt.savefig(hist_pdf)
    else:
        plt.show()
        plt.close()

    # show degree distribution
    # matplotlib is apparently segfaulting without a good reason
    cum_probabilities, bins = graph.degree_distribution()
    plt.figure()
    plt.plot(bins, cum_probabilities)
    plt.xscale("log")

    if file_prefix is not None:
        dist_pdf = file_prefix + "-dist.pdf"
        plt.savefig(dist_pdf)
    else:
        plt.show()
        plt.close()


def main():

    # either load the graph or mine it
    if args.load is not None:
        graph = PolarizationGraph.from_file(args.load)
    else:
        # mine data and store it
        contents = iter([])

        reddit_collector = RedditCollector()
        reddit_iter, users_flair = reddit_collector.collect(
            args.rn,
            args.r_kw,
            "AskTrumpSupporters",
            limit=args.rl,
            cross=args.rc,
        )

        contents = itertools.chain(contents, reddit_iter)

        graph = PolarizationGraph(contents, users_flair)

        if args.dump is not None:
            graph.dump(args.dump)

    if args.k > 0:
        graph.select_kcore(args.k)

    if args.stats_show or args.stats_save:
        compute_stats(graph, args.stats_save)

    if args.graph_draw_save is not None:
        graph.draw(output=args.graph_draw_save)
    elif (
        not args.graph_draw_no and not args.stats_show and not args.stats_save
    ):
        graph.draw()


if __name__ == "__main__":
    main()
