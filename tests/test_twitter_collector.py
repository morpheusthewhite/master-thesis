import treelib


from polarmine.collectors.twitter_collector import TwitterCollector
from polarmine.comment import Comment
from polarmine.thread import Thread


twitter_collector = TwitterCollector()


def test_twitter_find_statuses_easy():
    # verify that the collector finds the exact number of contents requested
    ncontents = 200
    statuses = twitter_collector.__find_statuses__(ncontents, None, "nytimes")
    assert len(statuses) == ncontents


def test_twitter_find_statuses_medium():
    # verify that the collector finds the exact number of contents requested
    ncontents = 600
    statuses = twitter_collector.__find_statuses__(ncontents, None, "nytimes")
    assert len(statuses) == ncontents


def test_twitter_find_statuses_hard():
    # verify that the collector finds the exact number of contents requested
    ncontents = 1000
    statuses = twitter_collector.__find_statuses__(ncontents, None, "nytimes")
    assert len(statuses) == ncontents


def test_twitter_collect_simple():
    # simple check on single content
    discussion_trees = list(
        twitter_collector.collect(1, keyword="obama", limit=10, cross=False)
    )
    assert len(discussion_trees) == 1

    discussion_tree = discussion_trees[0]
    assert isinstance(discussion_tree, treelib.Tree)


def test_twitter_collect_more():
    # try to collect more than 1 content
    discussion_trees = list(
        twitter_collector.collect(2, keyword="obama", limit=10, cross=False)
    )
    assert len(discussion_trees) == 2

    for discussion_tree in discussion_trees:
        assert isinstance(discussion_tree, treelib.Tree)

        root_id = discussion_tree.root
        root = discussion_tree[root_id]
        assert isinstance(root.data, Thread)

        children = discussion_tree.children(root_id)
        if len(children) > 0:
            assert isinstance(children[0].data, Comment)


def test_twitter_collect_page():
    # try to collect from page
    discussion_trees = list(
        twitter_collector.collect(2, page="Cristiano", limit=10, cross=True)
    )
    assert len(discussion_trees) >= 2


def test_twitter_shares():
    # try to collect status which have url shared by other
    # statuses
    discussion_trees = list(
        twitter_collector.collect(1, page="nytimes", limit=10, cross=True)
    )
    assert len(discussion_trees) >= 1

    content = None
    for discussion_tree in discussion_trees:
        assert isinstance(discussion_tree, treelib.Tree)

        root_id = discussion_tree.root
        root = discussion_tree[root_id]
        assert isinstance(root.data, Thread)

        children = discussion_tree.children(root_id)
        if len(children) > 0:
            assert isinstance(children[0].data, Comment)

        thread = root.data
        thread_content = thread.content
        if content is None:
            content = thread_content

        # all threads must have the same content
        assert thread_content == content
