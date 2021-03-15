from polarmine.collectors.reddit_collector import RedditCollector
from polarmine.collectors.twitter_collector import TwitterCollector
from polarmine.graph import PolarizationGraph
from polarmine.follow_graph import FollowGraph


reddit_collector = RedditCollector()
twitter_collector = TwitterCollector()


def test_follow_graph():
    follow_dict = {1: [0, 2, 3], 2: [0, 3], 3: []}

    follow_graph = FollowGraph(follow_dict, follow_dict)
    assert follow_graph is not None

    communities = list(follow_graph.communities())
    assert len(communities) == 3


def test_graph_construction_reddit_simple():
    contents, follow_dict = reddit_collector.collect(1, limit=10, cross=False)
    contents = list(contents)

    graph = PolarizationGraph(contents)
    assert graph is not None


def test_graph_construction_reddit_more():
    contents, follow_dict = reddit_collector.collect(4, limit=10)
    contents = list(contents)

    graph = PolarizationGraph(contents, follow_dict)
    assert graph is not None


def test_graph_construction_twitter_simple():
    contents, follow_dict = twitter_collector.collect(
        1, keyword="obama", limit=10, cross=False
    )
    contents = list(contents)

    graph = PolarizationGraph(contents, follow_dict)
    assert graph is not None


def test_graph_construction_twitter_more():
    contents, follow_dict = twitter_collector.collect(
        4, keyword="obama", limit=10, cross=False
    )
    contents = list(contents)

    graph = PolarizationGraph(contents, follow_dict)
    assert graph is not None


def test_graph_kcore_selection():
    contents, follow_dict = reddit_collector.collect(4, limit=10, cross=False)
    contents = list(contents)

    graph = PolarizationGraph(contents, follow_dict)
    num_vertices_unmasked = graph.graph.num_vertices()

    graph.select_kcore(2)
    num_vertices_masked = graph.graph.num_vertices()

    assert num_vertices_unmasked != num_vertices_masked
