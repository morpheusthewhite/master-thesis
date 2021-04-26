import pickle
import argparse

from polarmine.graph import PolarizationGraph


parser = argparse.ArgumentParser()
parser.add_argument(
    "--scores",
    "-sc",
    type=str,
    default=None,
    metavar="filename",
    dest="scores_filename",
    help="filename of scores pickle",
)
parser.add_argument(
    "--stats",
    "-st",
    type=str,
    default=None,
    metavar="filename",
    dest="stats_filename",
    help="filename of stats pickle",
)
parser.add_argument(
    "graph_filename",
    type=str,
    help="filename of graph",
)
parser.add_argument(
    "--alpha",
    "-a",
    type=float,
    default=0.4,
    metavar="value",
    help="value of alpha (used in thread/content controversy)",
)


MIP_KEY = "mip"
MIP_APPR_KEY = "mip_rounding_algorithm"
MIP_DENSEST_KEY = "mip-densest"


def analyze(
    graph_filename: str = None,
    scores_filename: str = None,
    stats_filename: str = None,
    alpha: float = 0.4,
):
    graph = PolarizationGraph.from_file(graph_filename)
    graph.remove_self_loops()

    if scores_filename is not None:
        with open(scores_filename, "rb") as scores_file:
            scores = pickle.load(scores_file)

        # visualize Echo Chamber solution
        if scores.get(MIP_KEY) is not None:
            graph.select_echo_chamber(alpha, scores[MIP_KEY][1])
            graph.draw()
            graph.clear_filters()

        # visualize Echo Chamber solution
        if scores.get(MIP_DENSEST_KEY) is not None:
            graph.select_echo_chamber(alpha, scores[MIP_APPR_KEY][1])
            graph.draw()
            graph.clear_filters()

        # visualize Echo Chamber approximation solution
        if scores.get(MIP_DENSEST_KEY) is not None:
            graph.select_echo_chamber(alpha, scores[MIP_DENSEST_KEY][1])
            graph.draw()
            graph.clear_filters()


if __name__ == "__main__":
    args = parser.parse_args()

    analyze(
        args.graph_filename,
        args.scores_filename,
        args.stats_filename,
        args.alpha,
    )
