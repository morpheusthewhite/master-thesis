import os

from polarmine.collectors.reddit_collector import RedditCollector
from polarmine.graph import PolarizationGraph


# folder containing the graph file
GRAPH_FOLDER = os.path.join("tests", "data")
# filename of the graph file
GRAPH_FILENAME = "test.gt"
# complete path to graph file
GRAPH_PATH = os.path.join(GRAPH_FOLDER, GRAPH_FILENAME)
N_CONTENTS = 4
reddit_collector = RedditCollector()


def test_graph_save():
    contents = list(
        reddit_collector.collect(
            N_CONTENTS, limit=10, page="programming", cross=False
        )
    )

    graph = PolarizationGraph(contents)
    assert graph is not None

    # create the folder containing the graph if it does not exist
    if not os.path.exists(GRAPH_FOLDER):
        os.mkdir(GRAPH_FOLDER)

    # save only if not existing (to speedup the tests)
    if not os.path.exists(GRAPH_PATH):
        graph.dump(GRAPH_PATH)


def test_graph_score_components():
    graph = PolarizationGraph.from_file(GRAPH_PATH)
    assert graph is not None

    graph.score_components(0.2)


def test_graph_score_greedy():
    graph = PolarizationGraph.from_file(GRAPH_PATH)
    assert graph is not None

    graph.score_greedy(0.1)