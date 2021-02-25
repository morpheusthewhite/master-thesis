import treelib


from polarmine.collectors.reddit_collector import RedditCollector
from polarmine.comment import Comment
from polarmine.content import Content


reddit_collector = RedditCollector()


def test_reddit_collect_simple():
    # simple check on single content
    contents, users = reddit_collector.collect(
        1, page="AskTrumpSupporters", limit=10, cross=False
    )
    contents = list(contents)
    assert len(contents) == 1

    content = contents[0]
    assert isinstance(content, treelib.Tree)


def test_reddit_collect_more():
    # try to collect more than 1 content
    threads, users = reddit_collector.collect(
        2, page="AskTrumpSupporters", limit=10, cross=False
    )
    threads = list(threads)
    assert len(threads) == 2

    for thread in threads:
        assert isinstance(thread, treelib.Tree)

        root_id = thread.root
        root = thread[root_id]
        assert isinstance(root.data, Content)

        children = thread.children(root_id)

        if len(children) > 0:
            assert isinstance(children[0].data, Comment)


def test_reddit_collect_page():
    # try to collect from page
    contents, users = reddit_collector.collect(
        2, page="AskTrumpSupporters", limit=10, cross=False
    )
    contents = list(contents)
    assert len(contents) == 2


#  def test_reddit_collect_page_cross():
#      # try to collect from page
#      contents = list(
#          reddit_collector.collect(
#              2, page="AskTrumpSupporters", limit=10, cross=True
#          )
#      )
#      assert len(contents) >= 2
#
#
#  def test_reddit_collect_keyword():
#      # try to collect from keyword
#      contents = list(reddit_collector.collect(2, keyword="obama", limit=10))
#      assert len(contents) >= 2
