import argparse
import itertools
import matplotlib.pyplot as plt

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
    "-s",
    "--stats",
    default=None,
    action="store_true",
    dest="stats",
    help="compute common graph statistics",
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

    graph.summarize()

    if args.stats:
        graph.remove_self_loops()

        print(f"Fraction of nodes in k-core: {graph.kcore_size()}")
        print(f"Fraction of negative edges: {graph.negative_edges_fraction()}")

        global_clustering, global_clustering_stddev = graph.global_clustering()
        print(
            f"Clustering coefficient: {global_clustering} with standard deviation {global_clustering_stddev}"
        )

        print(
            f"Average shortest path length: {graph.average_shortest_path_length()}"
        )

        # show degree histogram
        # matplotlib is apparently segfaulting without a good reason
        #  counts, bins = graph.degree_histogram_total()
        #  plt.figure()
        #  plt.plot(bins, counts)
        #
        #  plt.show()
        #  plt.close()

        # show degree distribution
        # matplotlib is apparently segfaulting without a good reason
        #  cum_probabilities, bins = graph.degree_distribution()
        #  plt.figure()
        #  plt.plot(bins, cum_probabilities)
        #  plt.xscale("log")
        #
        #  plt.show()
        #  plt.close()

    if args.graph_draw_save is not None:
        graph.draw(output=args.graph_draw_save)
    elif not args.graph_draw_no:
        graph.draw()


if __name__ == "__main__":
    main()
