import pickle
import argparse
import sys
import os

from polarmine.graph import PolarizationGraph

MIP_PARAM = "MIP"
ROUNDING_PARAM = "rounding"
MIP_DENSEST_PARAM = "MIP-densest"

MIP_KEY = "mip"
APPR_KEY = "mip_rounding_algorithm"
MIP_DENSEST_KEY = "mip-densest"


parser = argparse.ArgumentParser()
parser.add_argument(
    "--scores-filename",
    "-sf",
    type=str,
    default=None,
    metavar="filename",
    dest="scores_filename",
    help="filename of scores pickle",
)
parser.add_argument(
    "--score",
    "-sc",
    type=str,
    default=None,
    metavar="method",
    dest="score",
    help=f"the result of which score to analyze. Possible values: {MIP_PARAM}, {ROUNDING_PARAM}, {MIP_DENSEST_PARAM}",
)
#  parser.add_argument(
#      "--stats",
#      "-st",
#      type=str,
#      default=None,
#      metavar="filename",
#      dest="stats_filename",
#      help="filename of stats pickle containing the computed stats of the graph",
#  )
parser.add_argument(
    "--save-path",
    type=str,
    help="directory in which to save results. If not provided shows results",
    default=None,
    metavar="directory",
    dest="save_path",
)
parser.add_argument(
    "graph_filename", type=str, help="filename of graph", default=None
)
parser.add_argument(
    "--alpha",
    "-a",
    type=float,
    default=0.4,
    metavar="value",
    help="value of alpha (used in thread/content controversy)",
)


def analyze(
    graph_filename: str = None,
    scores_filename: str = None,
    #  stats_filename: str = None,
    save_path: str = None,
    score: str = None,
    alpha: float = 0.4,
):
    graph = PolarizationGraph.from_file(graph_filename)
    graph.remove_self_loops()

    #  if stats_filename is not None:
    #      stats_file = open(stats_filename, 'r')
    #  else:
    #      stats_file = sys.stdout

    if save_path is not None:
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        outfile = os.path.join(save_path, "results_stats.txt")
        graph_pdf_filename_echo_chambers = os.path.join(
            save_path, "graph_echo_chambers.pdf"
        )
        graph_pdf_filename_full = os.path.join(save_path, "graph_full.pdf")
    else:
        outfile = sys.stdout
        graph_pdf_filename_echo_chambers = None
        graph_pdf_filename_full = None

    if alpha == -1:
        alpha = graph.alpha_median()

    if scores_filename is not None:
        with open(scores_filename, "rb") as scores_file:
            scores = pickle.load(scores_file)

        # visualize Echo Chamber solution
        if score == MIP_PARAM:
            score_key = MIP_KEY
        elif score == ROUNDING_PARAM:
            score_key = APPR_KEY
        elif score == MIP_DENSEST_PARAM:
            score_key = MIP_DENSEST_KEY
        else:
            return
        score_key = f"{score_key}_{alpha}"

        users = scores[score_key][1]
        graph.select_echo_chamber(alpha, users)

        print(
            f"Number of vertices in the results: {graph.num_vertices()}",
            file=outfile,
        )
        print(
            f"Number of edges in the results: {graph.num_edges()}",
            file=outfile,
        )
        print(
            f"Number of components in the resulting graph: {graph.num_components()}",
            file=outfile,
        )

        graph.draw(output=graph_pdf_filename_echo_chambers)
        graph.clear_filters()

        print(
            f"Number of components in the original graph: {graph.num_components_from_vertices(users)}",
            file=outfile,
        )

        graph.draw(show_vertices=users, output=graph_pdf_filename_full)

        if save_path is not None:
            outfile.close()


if __name__ == "__main__":
    args = parser.parse_args()

    analyze(
        args.graph_filename,
        args.scores_filename,
        args.save_path,
        args.score,
        args.alpha,
    )
