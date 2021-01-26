import praw
import treelib
from typing import Optional

from polarmine.collectors.collector import Collector
from polarmine.content import Content


COMMENT_LIMIT = 10


class RedditCollector(Collector):
    """collects content from reddit
    """

    def __init__(self, **kwargs):
        super(RedditCollector, self).__init__(**kwargs)

        self.reddit = praw.Reddit()

    def __find_contents_id__(self, ncontents: int, keyword: Optional[str], page: Optional[str]) -> list[str]:
        """Find contents

        Args:
            ncontents (int): number of contents to return
            keyword (Optional[str]): keyword used for filtering content
            page (Optional[str]): the starting page from which content is
            found

        Returns:
            list[str]: the list of found praw submissions
        """
        if keyword is None:
            r_all = self.reddit.subreddit("all")
            contents_id = r_all.hot()
        else:
            # TODO
            pass

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

    def __submission_to_thread__(self, submission) -> treelib.Tree:
        """Use a submission to create the associated thread (of comments)

        Args:
            submission: the praw submission from which comments are extracted

        Returns:
            Tree: A Tree object associated to comments of the submission
            (which is the root)
        """
        # TODO: increase the limit?
        submission.comments.replace_more(limit=COMMENT_LIMIT)
        comment_forest = submission.comments

        thread = treelib.Tree()

        # the submission represents the root node in the tree collecting
        # all the replies

        # modify the id to follow convention user for `parent_id` attribute
        # of comment
        submission_id = f"t3_{submission.id}"

        thread.create_node(submission_id, submission_id)

        # iterate over comments to the submission
        for comment in comment_forest.list():

            # TODO: check MoreComment?
            #  if isinstance(comment, praw.models.MoreComments):
            #      continue
            #  else:

            # modify the id to follow convention user for `parent_id`
            id_ = f"t1_{comment.id}"
            parent = comment.parent_id

            author = comment.author
            text = comment.body
            # time is represent in UTC
            time = comment.created_utc
            # TODO: you should probably create a class for this
            data = {
                "author": author,
                "text": text,
                "timestamp": time
            }

            thread.create_node(id_, id_, parent, data)

        return thread

    def collect(self, ncontents, keyword=None, page=None) -> list[Content]:
        contents_id = self.__find_contents_id__(ncontents, keyword, page)

        for i, content_id in enumerate(contents_id):
            submission = self.reddit.submission(content_id)

            content = self.__submission_to_content__(submission, keyword)
            thread = self.__submission_to_thread__(submission)

            yield (content, thread)

            if i >= ncontents:
                break

