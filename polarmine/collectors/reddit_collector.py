import praw
import random
import treelib
import itertools
from typing import Optional

from polarmine.collectors.collector import Collector
from polarmine.thread import Thread
from polarmine.comment import Comment


class RedditCollector(Collector):
    """collects content from reddit"""

    def __init__(self, **kwargs):
        super(RedditCollector, self).__init__(**kwargs)

        self.reddit = praw.Reddit()

    def __find_contents_id__(
        self, ncontents: int, keyword: Optional[str], page: Optional[str]
    ) -> list[str]:
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
            contents_id = self.reddit.subreddit(page).hot(limit=ncontents)
        elif keyword is not None:
            contents_id = self.reddit.subreddit("all").search(
                keyword, limit=ncontents
            )
        else:
            contents_id = self.reddit.subreddit("all").hot(limit=ncontents)

        return contents_id

    def __submission_to_content__(
        self, submission: praw.models.Submission
    ) -> str:
        """Retrieve content url from a submission

        Args:
            submission: a praw submission model

        Returns:
            str: the url of the content (submission)
        """
        return submission.url

    def __submission_to_thread__(
        self, submission: praw.models.Submission, keyword: str, content: str
    ) -> Thread:
        """Create a Thread object from a submission

        Args:
            submission: a praw submission model
            keyword (str): keyword used for filtering submissions
            content (str): the url of the content

        Returns:
            Thread: the thread object relative to the submission
        """
        # check if author exist (the account may have been deleted)
        if submission.author is None:
            author_hash = hash(random.uniform(0, 1))
        else:
            author_hash = hash(submission.author.name)

        thread = Thread(
            submission.permalink,
            submission.title,
            submission.created_utc,
            author_hash,
            content,
            keyword,
        )

        return thread

    def __submission_to_discussion_trees__(
        self,
        submission: praw.models.Submission,
        keyword: str,
        limit: int,
        cross: bool,
    ) -> list[treelib.Tree]:
        """Use a submission to create the associated threads (of comments)

        Args:
            submission: the praw submission from which discussions are extracted
            keyword (str): keyword used for filtering submissions
            limit (int): maximum number of comments to unfold in the
                highest level
            cross (bool): if True looks for duplicates and return them as thread

        Returns:
            list(Tree): A list of Tree object associated to comments of the submission
                (which is the root). May contain more than 1 element if cross is True
        """
        # retrieve content object
        content = self.__submission_to_content__(submission)

        submissions = [submission]

        # eventually add duplicates/crossposts
        if cross:
            submissions.extend(submission.duplicates())

        discussion_trees = []

        # iterate over original submission and (eventually) crossposts
        for s in submissions:

            # modify the id to follow convention user for `parent_id` attribute
            # of comment
            s_id = f"t3_{s.id}"

            # retrieve comments
            s.comments.replace_more(limit)

            # notice the distinction between comment forest, which is a praw
            # model and discussion tree, which is an instance of treelib.Tree()
            comment_forest = s.comments

            # thread object, containing informations regarding this specific
            # submission
            thread = self.__submission_to_thread__(s, keyword, content)
            discussion_tree = treelib.Tree()

            # check user existance
            if s.author is None:
                author_hash = hash(random.uniform(0, 1))
            else:
                author_hash = hash(s.author.name)
            # the submission represents the root node in the tree collecting
            # all the replies. The associated data is a Thread object.
            # In this case the tag (submitter user) is the author of this
            # (possibly crossposted) submission and the id is the id
            # of this submission
            discussion_tree.create_node(
                tag=author_hash, identifier=s_id, data=thread
            )

            # iterate over comments to the submission
            for comment in comment_forest.list():

                # modify the id to follow convention user for `parent_id`
                id_ = f"t1_{comment.id}"
                parent = comment.parent_id

                # polarmine comment object, store minimal set of information
                if comment.author is None:
                    # comment has been removed, assing it to a user with
                    # random hash
                    author_hash = hash(random.uniform(0, 1))
                else:
                    author_hash = hash(comment.author.name)
                comment_polarmine = Comment(
                    comment.body, author_hash, comment.created_utc
                )

                discussion_tree.create_node(
                    tag=author_hash,
                    identifier=id_,
                    parent=parent,
                    data=comment_polarmine,
                )

            discussion_trees.append(discussion_tree)

        return discussion_trees

    def collect(
        self,
        ncontents: int,
        keyword: str = None,
        page: str = None,
        limit: int = 10000,
        cross: bool = True,
    ) -> list[treelib.Tree]:
        """collect content and their relative threads as tree.

        Args:
            ncontents: number of contents to find
            keyword (Optional[str]): keyword used for filtering content.
                If page is not None then it is ignored
            page (Optional[str]): the starting page from which content is
                found. A + separated list of subreddits (no space). For more
                info on the accepted format see
                https://praw.readthedocs.io/en/latest/code_overview/reddit_instance.html#praw.Reddit.subreddit
            limit (int): maximum number of comments to unfold in the
                highest level
            cross (bool): if True includes also crossposts of the found submissions

        Returns:
            list[Tree]: a list of tree, each associated to a submission.
                The root node is associated to the start of the thread
                itself and its `data`
                is a Content object, while for the other nodes it is a `Comment`
        """
        contents_id = self.__find_contents_id__(ncontents, keyword, page)
        discussion_trees = iter([])

        for i, content_id in enumerate(contents_id):
            submission = self.reddit.submission(content_id)

            content_discussion_trees = self.__submission_to_discussion_trees__(
                submission, keyword, limit, cross
            )
            discussion_trees = itertools.chain(
                discussion_trees, content_discussion_trees
            )

            if i + 1 >= ncontents:
                break

        return discussion_trees
