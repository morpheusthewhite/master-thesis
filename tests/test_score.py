import os

from polarmine.collectors.reddit_collector import RedditCollector
from polarmine.graph import PolarizationGraph
from polarmine.ecp import (
    score_from_vertices_index,
    ECPComponentsSolver,
    ECPMIPSolver,
    ECPRoundingSolver,
    ECPBetaSolver,
    ECPPeelingSolver,
)
from polarmine.decp import DECPMIPSolver


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

    ECPComponentsSolver().solve(graph, 0.2)


def test_graph_score_greedy_beta_positiveness():
    graph = PolarizationGraph.from_file(GRAPH_PATH)
    assert graph is not None

    ECPBetaSolver().solve(graph, 0.1)


def test_graph_score_greedy_beta_uniform():
    graph = PolarizationGraph.from_file(GRAPH_PATH)
    assert graph is not None

    ECPBetaSolver(positiveness_samples=False).solve(graph, 0.1)


def test_graph_score_greedy_peeling():
    graph = PolarizationGraph.from_file(GRAPH_PATH)
    assert graph is not None

    ECPPeelingSolver().solve(graph, 0.1)


def test_graph_score_mip():
    graph = PolarizationGraph.from_file(GRAPH_PATH)
    assert graph is not None
    graph.remove_self_loops()

    alpha = 0.3

    score, users, _, _ = ECPMIPSolver().solve(graph, alpha)
    score_vertices, _ = score_from_vertices_index(graph, users, alpha)
    assert score_vertices == score


def test_graph_score_mip_densest():
    graph = PolarizationGraph.from_file(GRAPH_PATH)
    assert graph is not None
    graph.remove_self_loops()

    alpha = 0.3
    score, users, _, _ = DECPMIPSolver().solve(graph, alpha)
    score_vertices, _ = score_from_vertices_index(graph, users, alpha)
    assert len(users) > 0
    assert score_vertices / float(len(users)) == score


def test_graph_score_mip_relaxation():
    graph = PolarizationGraph.from_file(GRAPH_PATH)
    assert graph is not None
    graph.remove_self_loops()

    alpha = 0.3
    ECPMIPSolver(relaxation=True).solve(graph, alpha)


def test_graph_score_rounding():
    graph = PolarizationGraph.from_file(GRAPH_PATH)
    assert graph is not None
    graph.remove_self_loops()

    ECPRoundingSolver().solve(graph, 0.1)


def test_graph_echo_chamber_selection():
    graph = PolarizationGraph.from_file(GRAPH_PATH)
    assert graph is not None
    graph.remove_self_loops()

    _, vertices, edges, _ = ECPMIPSolver().solve(graph, 0.4)
    graph.select_echo_chamber(0.4, vertices)

    assert graph.num_edges() == len(edges)


def test_graph_score_densest_nc_subgraph_simple():
    graph = PolarizationGraph.from_file(GRAPH_PATH)
    assert graph is not None
    graph.remove_self_loops()

    alpha = 0.4

    graph.score_densest_nc_subgraph(alpha, simple=True)


def test_graph_score_densest_nc_subgraph():
    graph = PolarizationGraph.from_file(GRAPH_PATH)
    assert graph is not None
    graph.remove_self_loops()

    alpha = 0.4

    graph.score_densest_nc_subgraph(alpha, simple=False)


def test_graph_score_o2_bff():
    graph = PolarizationGraph.from_file(GRAPH_PATH)
    assert graph is not None
    graph.remove_self_loops()

    alpha = 0.2

    score, vertices = graph.o2_bff_dcs_am(alpha, 2)
    assert score > 0
