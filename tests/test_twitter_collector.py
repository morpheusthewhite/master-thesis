import treelib


from polarmine.collectors.twitter_collector import TwitterCollector
from polarmine.comment import Comment
from polarmine.thread import Thread


twitter_collector = TwitterCollector()


def test_twitter_collect_simple():
    # simple check on single content
    discussion_trees, follow_dict = twitter_collector.collect(
        1, keyword="obama", limit=10, cross=False
    )
    discussion_trees = list(discussion_trees)

    assert len(discussion_trees) == 1

    discussion_tree = discussion_trees[0]
    assert isinstance(discussion_tree, treelib.Tree)

    assert len(list(follow_dict.keys())) > 0


def test_twitter_collect_more():
    # try to collect more than 1 content
    discussion_trees, follow_dict = twitter_collector.collect(
        2, keyword="obama", limit=10, cross=False
    )
    discussion_trees = list(discussion_trees)
    assert len(discussion_trees) == 2

    for discussion_tree in discussion_trees:
        assert isinstance(discussion_tree, treelib.Tree)

        root_id = discussion_tree.root
        root = discussion_tree[root_id]
        assert isinstance(root.data, Thread)

        children = discussion_tree.children(root_id)
        if len(children) > 0:
            assert isinstance(children[0].data, Comment)
            assert len(list(follow_dict.keys())) > 1

    assert len(list(follow_dict.keys())) > 0


def test_twitter_collect_page():
    # try to collect from page
    discussion_trees, follow_dict = twitter_collector.collect(
        2, page="Cristiano", limit=10, cross=True
    )
    discussion_trees = list(discussion_trees)

    assert len(discussion_trees) >= 2
    assert len(list(follow_dict.keys())) > 0


def test_twitter_shares():
    # try to collect status which have url shared by other
    # statuses
    discussion_trees, follow_dict = twitter_collector.collect(
        1, page="nytimes", limit=10, cross=True
    )
    discussion_trees = list(discussion_trees)
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
            assert len(list(follow_dict.keys())) > 1

        thread = root.data
        thread_content = thread.content
        if content is None:
            content = thread_content

        # all threads must have the same content
        assert thread_content == content

    assert len(list(follow_dict.keys())) > 0
