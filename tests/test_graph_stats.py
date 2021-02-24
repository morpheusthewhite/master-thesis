import os

from polarmine.collectors.reddit_collector import RedditCollector
from polarmine.graph import PolarizationGraph


# folder containing the graph file
GRAPH_FOLDER = os.path.join("tests", "data")
# filename of the graph file
GRAPH_FILENAME = "test.gt"
# complete path to graph file
GRAPH_PATH = os.path.join(GRAPH_FOLDER, GRAPH_FILENAME)
reddit_collector = RedditCollector()


# save a graph data if it does not exist
def test_graph_save():
    contents = list(
        reddit_collector.collect(4, limit=10, page="programming", cross=False)
    )

    graph = PolarizationGraph(contents)
    assert graph is not None

    # create the folder containing the graph if it does not exist
    if not os.path.exists(GRAPH_FOLDER):
        os.mkdir(GRAPH_FOLDER)

    # save only if not existing (to speedup the tests)
    if not os.path.exists(GRAPH_PATH):
        graph.dump(GRAPH_PATH)


def test_graph_load():
    graph = PolarizationGraph.from_file(GRAPH_PATH)
    assert graph is not None


def test_graph_negative():
    graph = PolarizationGraph.from_file(GRAPH_PATH)
    assert graph is not None

    graph.negative_edges_fraction()


def test_graph_negative_kcore():
    graph = PolarizationGraph.from_file(GRAPH_PATH)
    assert graph is not None

    graph.select_kcore(2)
    graph.negative_edges_fraction()


def test_graph_clustering():
    graph = PolarizationGraph.from_file(GRAPH_PATH)
    assert graph is not None

    graph.global_clustering()


def test_graph_clustering_kcore():
    graph = PolarizationGraph.from_file(GRAPH_PATH)
    assert graph is not None

    graph.select_kcore(2)
    graph.global_clustering()
