import argparse
import itertools
import matplotlib.pyplot as plt
import sys
import pickle

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
parser.add_argument(
    "-md",
    "--max-depth",
    type=int,
    default=4,
    metavar="d",
    dest="max_depth",
    help="Maximum thread depth when inferring user-content edges",
)
select_group = parser.add_mutually_exclusive_group()
select_group.add_argument(
    "-cu",
    "--select-content-user",
    action="store_true",
    default=False,
    dest="select_content_user",
    help="select and operate on the content-user subgraph",
)
select_group.add_argument(
    "-uu",
    "--select-user-user",
    action="store_true",
    default=False,
    dest="select_user_user",
    help="select and operate on the user-user subgraph",
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


def print_support_index_top_k(support_dict, file_, k=3):
    support_dict_sorted = {
        k: v for k, v in sorted(support_dict.items(), key=lambda item: item[1])
    }

    contents = list(support_dict_sorted.keys())
    k = min(k, len(contents))

    print(f"Lowest negative edge fraction top {k}:", file=file_)
    for i in range(k):
        content_ith = contents[i]

        print(
            f"\t{content_ith} with {support_dict_sorted[content_ith]}",
            file=file_,
        )

    print(f"Highest negative edge fraction top {k}:", file=file_)
    for i in range(1, k + 1):
        content_ith = contents[-i]

        print(
            f"\t{content_ith} with {support_dict_sorted[content_ith]}",
            file=file_,
        )


def print_accuracy_top_k(accuracy_dict, file_, k=6):
    support_dict_sorted = {
        k: v
        for k, v in sorted(accuracy_dict.items(), key=lambda item: item[1])
    }

    contents = list(support_dict_sorted.keys())
    k = min(k, len(contents))

    print(f"Highest accuracy top {k}:", file=file_)
    for i in range(1, k + 1):
        content_ith = contents[-i]

        print(
            f"\t{content_ith} with {accuracy_dict[content_ith]}",
            file=file_,
        )


def compute_stats(graph, file_prefix):
    # pickle results dictionary
    results = {}

    if file_prefix is None:
        stats_txt_file = sys.stdout
    else:
        stats_txt = file_prefix + "-stats.txt"
        stats_txt_file = open(stats_txt, "w")

    results["num_vertices"] = graph.num_vertices()
    results["num_edges"] = graph.num_edges()

    print(
        f'Number of vertices: {results["num_vertices"]}', file=stats_txt_file
    )
    print(f'Number of edges: {results["num_edges"]}', file=stats_txt_file)

    counts, bins = graph.support_index_histogram()
    results["support_histogram"] = (counts, bins)

    plt.figure()
    plt.plot(bins, counts)

    if file_prefix is not None:
        dist_pdf = file_prefix + "-support-hist.pdf"
        plt.savefig(dist_pdf)
    else:
        plt.show()
        plt.close()

    support_dict = graph.support_index_dict()
    results["support_dict"] = support_dict
    print_support_index_top_k(support_dict, stats_txt_file)

    accuracy_dict = graph.content_classification_accuracy()
    results["accuracy_dict"] = accuracy_dict

    plt.figure()
    plt.hist(list(accuracy_dict.values()))

    if file_prefix is not None:
        accuracy_hist_pdf = file_prefix + "-accuracy-hist.pdf"
        plt.savefig(accuracy_hist_pdf)
    else:
        plt.show()
        plt.close()

    print_accuracy_top_k(accuracy_dict, stats_txt_file)

    social_balance_accuracy = graph.social_balance_accuracy()
    print(
        f"Accuracy of social balance classification: {social_balance_accuracy}",
        file=stats_txt_file,
    )

    if file_prefix is not None:
        stats_txt_file.close()

        with open(file_prefix + "-results.p", "wb") as pickle_file:
            pickle.dump(results, pickle_file)


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

    graph.select_thread_depth(args.max_depth)

    if args.select_content_user:
        graph.select_content_user()
    elif args.select_user_user:
        graph.select_user_user()

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
