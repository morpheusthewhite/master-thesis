from polarmine.collectors.reddit_collector import RedditCollector
from polarmine.collectors.twitter_collector import TwitterCollector
from polarmine.graph import PolarizationGraph


reddit_collector = RedditCollector()
twitter_collector = TwitterCollector()


def test_graph_construction_reddit_simple():
    contents, users_flair = reddit_collector.collect(
        1, page="AskTrumpSupporters", limit=10, cross=False
    )

    graph = PolarizationGraph(contents, users_flair)
    assert graph is not None

    accuracy_dict = graph.content_classification_accuracy()
    assert accuracy_dict is not None

    # top level comments
    graph.select_thread_depth(0)

    # more comments
    graph.select_thread_depth(2)


def test_graph_construction_reddit_more():
    contents, users_flair = reddit_collector.collect(
        4, page="AskTrumpSupporters", limit=10
    )

    graph = PolarizationGraph(contents, users_flair)
    assert graph is not None


#  def test_graph_construction_twitter_simple():
#      contents = list(
#          twitter_collector.collect(1, keyword="obama", limit=10, cross=False)
#      )
#
#      graph = PolarizationGraph(contents)
#      assert graph is not None
#
#
#  def test_graph_construction_twitter_more():
#      contents = list(
#          twitter_collector.collect(4, keyword="obama", limit=10, cross=False)
#      )
#
#      graph = PolarizationGraph(contents)
#      assert graph is not None
#
#
#  def test_graph_kcore_selection():
#      contents, users_flair = list(
#          reddit_collector.collect(4, limit=10, cross=False)
#      )
#
#      graph = PolarizationGraph(contents, users_flair)
#      num_vertices_unmasked = graph.graph.num_vertices()
#
#      graph.select_kcore(2)
#      num_vertices_masked = graph.graph.num_vertices()
#
#      assert num_vertices_unmasked != num_vertices_masked
