import treelib


from polarmine.collectors.twitter_collector import TwitterCollector
from polarmine.comment import Comment
from polarmine.content import Content


twitter_collector = TwitterCollector()


def test_twitter_collect_simple():
    # simple check on single content
    contents = list(twitter_collector.collect(
        1, keyword="obama", limit=10, cross=False))
    assert len(contents) == 1

    content = contents[0]
    assert isinstance(content, treelib.Tree)


def test_twitter_collect_more():
    # try to collect more than 1 content
    threads = list(twitter_collector.collect(
        2, keyword="obama", limit=10, cross=False))
    assert len(threads) == 2

    for thread in threads:
        assert isinstance(thread, treelib.Tree)

        root_id = thread.root
        root = thread[root_id]
        assert isinstance(root.data, Content)

        children = thread.children(root_id)
        if len(children) > 0:
            assert isinstance(children[0].data, Comment)


def test_twitter_collect_page():
    # try to collect from page
    contents = list(twitter_collector.collect(
        2, page="Cristiano", limit=10, cross=True))
    assert len(contents) >= 2


def test_twitter_shares():
    # try to collect status which have url shared by other
    # statuses
    contents = list(twitter_collector.collect(
        2, page="nytimes", limit=10, cross=True))
    assert len(contents) >= 2
