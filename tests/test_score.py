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


def test_graph_score_greedy_beta_positiveness():
    graph = PolarizationGraph.from_file(GRAPH_PATH)
    assert graph is not None

    graph.score_greedy_beta(0.1)


def test_graph_score_greedy_beta_uniform():
    graph = PolarizationGraph.from_file(GRAPH_PATH)
    assert graph is not None

    graph.score_greedy_beta(0.1, positiveness_samples=False)


def test_graph_score_greedy_peeling():
    graph = PolarizationGraph.from_file(GRAPH_PATH)
    assert graph is not None

    graph.score_greedy_peeling(0.1)


def test_graph_score_mip():
    graph = PolarizationGraph.from_file(GRAPH_PATH)
    assert graph is not None
    graph.remove_self_loops()

    alpha = 0.3
    score, users, _, _ = graph.score_mip(alpha)
    score_vertices, _ = graph.score_from_vertices_index(users, alpha)
    assert score_vertices == score


def test_graph_score_mip_densest():
    graph = PolarizationGraph.from_file(GRAPH_PATH)
    assert graph is not None
    graph.remove_self_loops()

    alpha = 0.3
    score, users, _, _ = graph.score_mip_densest(alpha)
    #  score_vertices, _ = graph.score_from_vertices_index(users, alpha)
    #  assert score_vertices == score


def test_graph_score_mip_relaxation():
    graph = PolarizationGraph.from_file(GRAPH_PATH)
    assert graph is not None
    graph.remove_self_loops()

    graph.score_mip(0.1, relaxation=True)


def test_graph_score_mip_relaxation_r():
    graph = PolarizationGraph.from_file(GRAPH_PATH)
    assert graph is not None
    graph.remove_self_loops()

    graph.score_relaxation_algorithm(0.1)


def test_graph_echo_chamber_selection():
    graph = PolarizationGraph.from_file(GRAPH_PATH)
    assert graph is not None
    graph.remove_self_loops()

    _, vertices, edges, _ = graph.score_mip(0.4)
    graph.select_echo_chamber(0.4, vertices)

    assert graph.num_edges() == len(edges)


def test_graph_score_densest_nc_subgraph():
    graph = PolarizationGraph.from_file(GRAPH_PATH)
    assert graph is not None
    graph.remove_self_loops()

    alpha = 0.4

    graph.score_densest_nc_subgraph(alpha)
