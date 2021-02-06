from polarmine.collectors.reddit_collector import RedditCollector
from polarmine.collectors.twitter_collector import TwitterCollector
from polarmine.graph import PolarizationGraph


reddit_collector = RedditCollector()
twitter_collector = TwitterCollector()


def test_graph_construction_reddit_simple():
    contents = list(reddit_collector.collect(1, limit=10))

    graph = PolarizationGraph(contents)
    assert graph is not None


def test_graph_construction_reddit_more():
    contents = list(reddit_collector.collect(4, limit=10))

    graph = PolarizationGraph(contents)
    assert graph is not None


def test_graph_construction_twitter_simple():
    contents = list(twitter_collector.collect(1, keyword="obama",
                                              limit=10))

    graph = PolarizationGraph(contents)
    assert graph is not None


def test_graph_construction_twitter_more():
    contents = list(twitter_collector.collect(4, keyword="obama",
                                              limit=10))

    graph = PolarizationGraph(contents)
    assert graph is not None

