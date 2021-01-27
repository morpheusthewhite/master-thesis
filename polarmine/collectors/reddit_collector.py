import praw
import treelib
from typing import Optional

from polarmine.collectors.collector import Collector
from polarmine.content import Content
from polarmine.comment import Comment


COMMENT_LIMIT = 10


class RedditCollector(Collector):
    """collects content from reddit
    """

    def __init__(self, **kwargs):
        super(RedditCollector, self).__init__(**kwargs)

        self.reddit = praw.Reddit()

    def __find_contents_id__(self, ncontents: int, keyword: Optional[str],
                             page: Optional[str]) -> list[str]:
        """Find contents

        Args:
            ncontents (int): number of contents to return
            keyword (Optional[str]): keyword used for filtering content.
                If page is not None then it is ignored
            page (Optional[str]): the starting page from which content is
                found. A + separated list of subreddits (no space). For more
                info on the accepted format see
                https://praw.readthedocs.io/en/latest/code_overview/reddit_instance.html#praw.Reddit.subreddit

        Returns:
            list[str]: the list of found praw submissions
        """
        if page is not None:
            contents_id = self.reddit.subreddit(page).hot()
        elif keyword is not None:
            contents_id = self.reddit.subreddit("all").search(keyword)
        else:
            contents_id = self.reddit.subreddit("all").hot()

        return contents_id

    def __submission_to_content__(self, submission, keyword: str) -> Content:
        """Create a Content object from a submission

        Args:
            submission: a praw submission model
            keyword (str): keyword used for filtering submissions

        Returns:
            Content: the content object relative to the submission
        """
        # TODO: just consider the title?
        content = Content(submission.url, submission.title,
                          submission.created_utc,
                          submission.author, keyword)

        return content

    def __submission_to_thread__(self, submission, keyword) -> treelib.Tree:
        """Use a submission to create the associated thread (of comments)

        Args:
            submission: the praw submission from which comments are extracted
            keyword (str): keyword used for filtering submissions

        Returns:
            Tree: A Tree object associated to comments of the submission
                (which is the root)
        """
        # retrieve content object
        content = self.__submission_to_content__(submission, keyword)

        # modify the id to follow convention user for `parent_id` attribute
        # of comment
        submission_id = f"t3_{submission.id}"

        # retrieve comments
        # TODO: increase the limit?
        submission.comments.replace_more(limit=COMMENT_LIMIT)
        comment_forest = submission.comments

        thread = treelib.Tree()

        # the submission represents the root node in the tree collecting
        # all the replies. The associated data is a content object
        thread.create_node(submission_id, submission_id, data=content)

        # iterate over comments to the submission
        for comment in comment_forest.list():

            # modify the id to follow convention user for `parent_id`
            id_ = f"t1_{comment.id}"
            parent = comment.parent_id

            # polarmine comment object, store minimal set of information
            comment_pm = Comment(comment.body, comment.author,
                                 comment.created_utc)

            thread.create_node(id_, id_, parent, data=comment_pm)

        return thread

    def collect(self, ncontents, keyword=None, page=None) \
            -> list[treelib.Tree]:
        """collect content and their relative comments as tree.

        Args:
            ncontents: number of submission to find
            keyword (Optional[str]): keyword used for filtering content.
                If page is not None then it is ignored
            page (Optional[str]): the starting page from which content is
                found. A + separated list of subreddits (no space). For more
                info on the accepted format see
                https://praw.readthedocs.io/en/latest/code_overview/reddit_instance.html#praw.Reddit.subreddit

        Returns:
            list[Tree]: a list of tree, each associated to a submission.
                The root node is associated to the content itself and its `data`
                is a Content object, while for the other nodes it is a `Comment`
        """
        contents_id = self.__find_contents_id__(ncontents, keyword, page)

        for i, content_id in enumerate(contents_id):
            submission = self.reddit.submission(content_id)

            thread = self.__submission_to_thread__(submission, keyword)

            yield (thread)

            if i + 1 >= ncontents:
                break

