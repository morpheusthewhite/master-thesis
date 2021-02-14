import os
import math
import treelib
import tweepy
import itertools
import urllib
import validators
from typing import Optional, Iterator

from polarmine.collectors.collector import Collector
from polarmine.content import Content
from polarmine.comment import Comment
from polarmine.tweepy import APIv2


QUOTE_MIN_REPLIES = 1
TWEET_MIN_REPLIES = 1


class TwitterCollector(Collector):
    """TwitterCollector."""

    def __init__(self, **kwargs):
        super(TwitterCollector, self).__init__(**kwargs)

        (
            consumer_key,
            consumer_secret,
            access_key,
            access_secret,
        ) = self.__get_keys__()

        # authorize twitter, initialize tweepy
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_key, access_secret)
        self.twitter = APIv2(auth, wait_on_rate_limit=True)

    def __get_keys__(self):
        """Retrieve twitter keys from environment"""
        consumer_key = os.getenv("TWITTER_CONSUMER_KEY")
        consumer_secret = os.getenv("TWITTER_CONSUMER_SECRET")

        access_key = os.getenv("TWITTER_ACCESS_KEY")
        access_secret = os.getenv("TWITTER_ACCESS_SECRET")

        if (
            consumer_secret is None
            or consumer_key is None
            or access_key is None
            or access_secret is None
        ):
            raise Exception(
                "You didn't properly setup twitter \
                environment variable, follow the README"
            )

        return consumer_key, consumer_secret, access_key, access_secret

    def __find_statuses__(
        self,
        ncontents: int,
        keyword: Optional[str],
        page: Optional[str],
    ) -> list[tweepy.Status]:
        """Find n statuses containing `keyword` or from a certain user `page`.
        Either `keyword` or `page` must not be None

        Args:
            ncontents (int): the number of statuses to find
            keyword (Optional[str]): the keyword used to filter tweets
            page (Optional[str]): the user from which tweets are retrieved

        Returns:
            list[tweepy.Status]: a list of statuses
        """
        if keyword is not None:
            cursor = tweepy.Cursor(
                self.twitter.search,
                q=f"{keyword} min_replies:{TWEET_MIN_REPLIES} -filter:replies",
                tweet_mode="extended",
            )
        elif page is not None:
            cursor = tweepy.Cursor(
                self.twitter.user_timeline,
                screen_name=page,
                tweet_mode="extended",
                exclude_replies=True,
            )
        else:
            raise NotImplementedError

        return list(cursor.items(ncontents))

    def __status_to_shares__(self, status: tweepy.Status) -> Iterator:
        """Find statuses which sharing the same external url of the given status

        Args:
            status (tweepy.Status): status from which external url is extracted, if present

        Returns:
            Iterator: an iterator over statuses sharing the same url
        """
        # check if url is present in the provided status (it is supposed to be at the end)
        url = status.full_text.split()[-1]

        # check if it is a valid url
        if validators.url(url):
            try:
                url_redirected = urllib.request.urlopen(url).url

            except urllib.error.HTTPError:
                # generic error, happens when tweet has some images?
                return iter([])

            url_parsed = urllib.parse.urlparse(url_redirected)

            # remove query parameters from the url
            url_cleaned = urllib.parse.urljoin(url_redirected, url_parsed.path)

            query = f"{url_cleaned} min_replies:{TWEET_MIN_REPLIES}"
            cursor = tweepy.Cursor(
                self.twitter.search, q=query, tweet_mode="extended"
            )

            return cursor.items()
        else:
            return iter([])

    def __status_to_thread_aux__(
        self, status: tweepy.Status, root_data: dict = None, limit: int = 10000
    ) -> treelib.Tree:
        """Find thread of comments associated to a certain status

        Args:
            status (tweepy.Status): the status for which replies are looked for
            root_data (dict): if not None, it is used as `data` for the root
            node of the resulting thread, otherwise a Comment object is
            attached
            limit (int): maximum number of tweets to check when looking
            for replies

        Returns:
            treelib.Tree: the Tree of comment replies
        """
        status_author_name = status.author.screen_name
        status_id = status.id

        # tree object storing comment thread
        thread = treelib.Tree()

        if root_data is not None:
            # use provided data
            thread.create_node(
                tag=hash(status_author_name),
                identifier=status_id,
                data=root_data,
            )
        else:
            # create comment object, associated to root node of this tree
            # the tag of the node is the author of the tweet
            comment_text = status.full_text
            comment_author = hash(status.author.screen_name)
            comment_time = status.created_at.timestamp()
            comment = Comment(comment_text, comment_author, comment_time)
            thread.create_node(
                tag=comment.author, identifier=status_id, data=comment
            )

        # get dictionary of replies
        # where each key is the str(id) of a tweet and the value is the
        # list of id of the replies to it
        replies_dict = self.twitter.get_replies_id(
            status_id, status_author_name
        )

        # initially the queue will contain only the children of the root node
        queue = replies_dict[status_id]

        while len(queue) > 0:
            reply = queue.pop(0)

            # replies to the current reply
            reply_replies = replies_dict.get(reply, [])

            # require 100 tweets at a time
            for i in range(math.ceil((len(reply_replies)) // 100)):

                # probably needs int instead of string
                statuses_batch = self.twitter.statuses_lookup(
                    id_=reply_replies[i * 100 : (i + 1) * 100]
                )

                for s in statuses_batch:
                    # create comment object, associated to root node of this tree
                    # the tag of the node is the author of the tweet
                    comment_text = status.full_text
                    comment_author = hash(status.author.screen_name)
                    comment_time = status.created_at.timestamp()
                    comment = Comment(
                        comment_text, comment_author, comment_time
                    )
                    thread.create_node(
                        tag=comment.author,
                        identifier=status_id,
                        data=comment,
                        parent=reply,
                    )

                    queue.extend(replies_dict[s.id_str])

        return thread

    def __status_to_thread__(
        self, status: tweepy.Status, keyword: str, limit: int, cross: bool
    ) -> treelib.Tree:
        """Find thread of comments associated to a certain status

        Args:
            status (tweepy.Status): the status for which reply are looked for
            keyword (str): the keyword used to filter status
            limit (int): maximum number of tweets to check when looking
            for replies

        Returns:
            treelib.Tree: the Tree of comment replies. The root node,
            corresponding to the status itself, is associated with a
            `Content` object in the node `data` while the other node have
            a `Comment` object
        """
        status_author_name = status.author.screen_name
        status_id = status.id

        # create content object, associated to root node
        content_url = f"https://twitter.com/user/status/{status_id}"
        content_text = status.full_text
        content_time = status.created_at.timestamp()
        content_author = hash(status_author_name)
        content = Content(
            content_url, content_text, content_time, content_author, keyword
        )

        # thread/tree including only replies to original status
        thread = self.__status_to_thread_aux__(status, content, limit)

        if cross:
            # add quote tweets of the obtained tweets
            # for each tweet search the twitter url, requiring at least
            # QUOTE_MIN_REPLIES reply
            query = f"https://twitter.com/{status_author_name}/status/{status_id} min_replies:{QUOTE_MIN_REPLIES}"

            # cursor over quotes of the status
            cursor_quote = tweepy.Cursor(
                self.twitter.search, q=query, tweet_mode="extended"
            )

            for quote_reply in cursor_quote.items():
                # quote replies can be handled as normal status since their
                # text is the reply (without including the quote)
                subthread = self.__status_to_thread_aux__(
                    quote_reply, limit=limit
                )

                # add subthread as children of the root
                thread.paste(status_id, subthread)

            for status_share in self.__status_to_shares__(status):
                # create content object, associated to root node
                content_share_url = (
                    f"https://twitter.com/user/status/{status_share.id}"
                )
                content_share_text = status_share.full_text
                content_share_time = status_share.created_at.timestamp()
                content_share_author = hash(status_share.author.screen_name)
                content_share = Content(
                    content_share_url,
                    content_share_text,
                    content_share_time,
                    content_share_author,
                    keyword,
                )

                subthread = self.__status_to_thread_aux__(
                    status_share, limit=limit, root_data=content_share
                )

                # TODO: add subthread as children of the root?
                #  thread.paste(status_id, subthread)
                yield subthread

        yield thread

    def collect(
        self,
        ncontents: int,
        keyword: str = None,
        page: str = None,
        limit: int = 10000,
        cross: bool = True,
    ) -> list[Content]:
        """collect content and their relative comments as tree.

        Args:
            ncontents: number of posts to find
            keyword (Optional[str]): keyword used for filtering content.
            If page is not None then it is ignored
            page (Optional[str]): the starting page from which content is
            found.
            limit (int): maximum number of tweets to check when looking
            for replies
            cross (bool): if True includes also the retweets of the found statuses
            in the result

        Returns:
            list[Tree]: a list of tree, each associated to a post.
            The root node is associated to the content itself and its `data`
            is a Content object, while for the other nodes it is a `Comment`
        """
        statuses = self.__find_statuses__(ncontents, keyword, page)
        threads = iter([])

        for status in statuses:

            content_threads = self.__status_to_thread__(
                status, keyword, limit, cross
            )
            threads = itertools.chain(threads, content_threads)

        return threads
