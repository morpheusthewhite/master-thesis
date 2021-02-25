import praw
import treelib
import itertools
from typing import Optional

from polarmine.collectors.collector import Collector
from polarmine.content import Content
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
            contents_id = self.reddit.subreddit(page).hot()
        elif keyword is not None:
            contents_id = self.reddit.subreddit("all").search(keyword)
        else:
            contents_id = self.reddit.subreddit("all").hot()

        return contents_id

    def __submission_to_content__(
        self, submission: praw.models.Submission, keyword: str
    ) -> Content:
        """Create a Content object from a submission

        Args:
            submission: a praw submission model
            keyword (str): keyword used for filtering submissions

        Returns:
            Content: the content object relative to the submission
        """
        # TODO: just consider the title?
        content = Content(
            submission.url,
            submission.title,
            submission.created_utc,
            hash(submission.author.name),
            keyword,
        )

        return content

    def __submission_to_thread__(
        self,
        submission: praw.models.Submission,
        keyword: str,
        limit: int,
        cross: bool,
    ) -> list[treelib.Tree]:
        """Use a submission to create the associated thread (of comments)

        Args:
            submission: the praw submission from which comments are extracted
            keyword (str): keyword used for filtering submissions
            limit (int): maximum number of comments to unfold in the
                highest level
            cross (bool): if True looks for duplicate and return their threads

        Returns:
            list(Tree): A list of Tree object associated to comments of the submission
                (which is the root). May contain more than 1 element if cross is True
        """
        # retrieve content object
        content = self.__submission_to_content__(submission, keyword)

        submissions = [submission]

        # eventually add duplicates/crossposts
        if cross:
            submissions.extend(submission.duplicates())

        threads = []
        # dictionary with key being the hash of the user id and value the flair
        users_flair = {}

        for s in submissions:
            # modify the id to follow convention user for `parent_id` attribute
            # of comment
            s_id = f"t3_{s.id}"

            # retrieve comments
            s.comments.replace_more(limit)
            comment_forest = s.comments

            thread = treelib.Tree()

            # the submission represents the root node in the tree collecting
            # all the replies. The associated data is a content object
            # in this case the tag (submitter user) is the author of this
            # (possibly crossposted) submission and the id is the id of the new
            # submission
            author_hash = hash(s.author.name)
            thread.create_node(tag=author_hash, identifier=s_id, data=content)
            users_flair[author_hash] = s.author_flair_text

            # iterate over comments to the submission
            for comment in comment_forest.list():

                # modify the id to follow convention user for `parent_id`
                id_ = f"t1_{comment.id}"
                parent = comment.parent_id

                # polarmine comment object, store minimal set of information
                author_hash = hash(comment.author)
                comment_pm = Comment(
                    comment.body, author_hash, comment.created_utc
                )
                users_flair[author_hash] = comment.author_flair_text

                thread.create_node(
                    tag=author_hash,
                    identifier=id_,
                    parent=parent,
                    data=comment_pm,
                )

            threads.append(thread)

        return threads, users_flair

    def collect(
        self,
        ncontents: int,
        keyword: str = None,
        page: str = None,
        limit: int = 10000,
        cross: bool = True,
    ) -> list[treelib.Tree]:
        """collect content and their relative comments as tree.

        Args:
            ncontents: number of submission to find
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
                The root node is associated to the content itself and its `data`
                is a Content object, while for the other nodes it is a `Comment`
        """
        contents_id = self.__find_contents_id__(ncontents, keyword, page)
        threads = iter([])
        users_flair_aggregated = {}

        for i, content_id in enumerate(contents_id):
            submission = self.reddit.submission(content_id)

            content_threads, users_flair = self.__submission_to_thread__(
                submission, keyword, limit, cross
            )
            threads = itertools.chain(threads, content_threads)

            for user, flair in users_flair.items():
                users_flair_aggregated[user] = flair

            if i + 1 >= ncontents:
                break

        return threads, users_flair_aggregated
