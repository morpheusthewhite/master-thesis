import numpy as np

from polarmine.collectors.reddit_collector import RedditCollector
from polarmine.collectors.twitter_collector import TwitterCollector
from polarmine.graph import PolarizationGraph


reddit_collector = RedditCollector()
twitter_collector = TwitterCollector()


def test_graph_construction_reddit_simple():
    contents = list(reddit_collector.collect(1, limit=10, cross=False))

    graph = PolarizationGraph(contents)
    assert graph is not None


def test_graph_construction_reddit_more():
    contents = list(reddit_collector.collect(4, limit=10))

    graph = PolarizationGraph(contents)
    assert graph is not None


def test_graph_construction_twitter_simple():
    contents = list(
        twitter_collector.collect(1, keyword="obama", limit=10, cross=False)
    )

    graph = PolarizationGraph(contents)
    assert graph is not None


def test_graph_construction_twitter_more():
    contents = list(
        twitter_collector.collect(4, keyword="obama", limit=10, cross=False)
    )

    graph = PolarizationGraph(contents)
    assert graph is not None


def test_graph_kcore_selection():
    contents = list(reddit_collector.collect(4, limit=10, cross=False))

    graph = PolarizationGraph(contents)
    num_vertices_unmasked = graph.graph.num_vertices()

    graph.select_kcore(2)
    num_vertices_masked = graph.graph.num_vertices()

    assert num_vertices_unmasked != num_vertices_masked


def test_model_generation():
    n_nodes = [20, 10]
    omega_positive = [[0.6, 0.1], [0.2, 0.7]]
    omega_negative = [[0.2, 0.4], [0.4, 0.1]]

    graph = PolarizationGraph.from_model(
        np.array(n_nodes),
        2,
        np.array(omega_positive),
        np.array(omega_negative),
    )

    assert graph.num_vertices() == 30
