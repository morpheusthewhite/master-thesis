import treelib


from polarmine.collectors.reddit_collector import RedditCollector
from polarmine.comment import Comment
from polarmine.thread import Thread


reddit_collector = RedditCollector()


def test_reddit_collect_simple():
    # simple check on single content
    discussion_trees = list(reddit_collector.collect(1, limit=10, cross=False))
    assert len(discussion_trees) == 1

    content = discussion_trees[0]
    assert isinstance(content, treelib.Tree)


def test_reddit_collect_more():
    # try to collect more than 1 content
    discussion_trees = list(reddit_collector.collect(2, limit=10, cross=False))
    assert len(discussion_trees) == 2

    for discussion_tree in discussion_trees:
        assert isinstance(discussion_tree, treelib.Tree)

        root_id = discussion_tree.root
        root = discussion_tree[root_id]
        assert isinstance(root.data, Thread)

        children = discussion_tree.children(root_id)

        if len(children) > 0:
            assert isinstance(children[0].data, Comment)


def test_reddit_collect_page():
    # try to collect from page
    discussion_trees = list(
        reddit_collector.collect(2, page="programming", limit=10, cross=False)
    )
    assert len(discussion_trees) == 2


def test_reddit_collect_page_cross():
    # try to collect from page
    discussion_trees = list(
        reddit_collector.collect(2, page="programming", limit=10, cross=True)
    )
    assert len(discussion_trees) >= 2


def test_reddit_collect_keyword():
    # try to collect from keyword
    discussion_trees = list(
        reddit_collector.collect(2, keyword="obama", limit=10)
    )
    assert len(discussion_trees) >= 2
