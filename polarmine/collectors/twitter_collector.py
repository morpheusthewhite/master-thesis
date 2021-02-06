import os
import time
import treelib
import tweepy
import itertools
from typing import Optional

from polarmine.collectors.collector import Collector
from polarmine.content import Content
from polarmine.comment import Comment


QUOTE_MIN_REPLIES = 1


class TwitterCollector(Collector):
    def __init__(self, **kwargs):
        super(TwitterCollector, self).__init__(**kwargs)

        consumer_key, consumer_secret, auth_key, auth_secret = self.__get_keys__()

        # authorize twitter, initialize tweepy
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(auth_key, auth_secret)
        self.twitter = tweepy.API(auth, wait_on_rate_limit=True)

    def __get_keys__(self):
        """Retrieve twitter keys from environment"""
        consumer_key = os.getenv("TWITTER_CONSUMER_KEY")
        consumer_secret = os.getenv("TWITTER_CONSUMER_SECRET")

        auth_key = os.getenv("TWITTER_AUTH_KEY")
        auth_secret = os.getenv("TWITTER_AUTH_SECRET")

        if (
            consumer_secret is None
            or consumer_key is None
            or auth_key is None
            or auth_secret is None
        ):
            raise Exception(
                "You didn't properly setup twitter \
                environment variable, follow the README"
            )

        return consumer_key, consumer_secret, auth_key, auth_secret

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
                self.twitter.search, q=keyword, tweet_mode="extended"
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

    def __reply_to_thread__(
        self, reply: tweepy.Status, limit: int = 10000
    ) -> treelib.Tree:
        """Find thread of comments associated to a certain reply

        Args:
            reply (tweepy.Status): the status for which reply are looked for
            limit (int): maximum number of tweets to check when looking
            for replies

        Returns:
            treelib.Tree: the Tree of comment replies
        """
        reply_author_name = reply.author.screen_name
        reply_id = reply.id

        # tree object storing comment thread
        thread = treelib.Tree()

        # create comment object, associated to root node of this tree
        # the tag of the node is the author of the tweet
        # (or retweet, eventually)
        comment_text = reply.full_text
        comment_author = hash(reply.author.screen_name)
        comment_time = reply.created_at.timestamp()
        comment = Comment(comment_text, comment_author, comment_time)
        thread.create_node(
            tag=comment.author, identifier=reply_id, data=comment
        )

        # cursor over replies to tweet
        replies = tweepy.Cursor(
            self.twitter.search,
            q=f"to:{reply_author_name}",
            since_id=reply_id,
            tweet_mode="extended",
        ).items()

        for i in range(limit):
            try:
                reply = replies.next()

                if (
                    reply.in_reply_to_status_id is not None
                    and reply.in_reply_to_status_id == reply_id
                ):

                    # obtain thread originated from current reply
                    subthread = self.__reply_to_thread__(reply, limit)

                    # add subthread as children of the current node
                    thread.paste(reply_id, subthread)

            except tweepy.RateLimitError:
                print("Twitter api rate limit reached")
                time.sleep(60)
                continue

            except StopIteration:
                break

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

        # list of statuses to be processes. Eventually includes quoting statuses
        statuses = [status]

        if cross:
            # add quote tweets of the obtained tweets
            # for each tweet search the twitter url, requiring at least
            # QUOTE_MIN_REPLIES reply
            query = f"https://twitter.com/{status_author_name}/status/{status_id} min_replies:{QUOTE_MIN_REPLIES}"

            # cursor over quotes of the status
            cursor_quote = tweepy.Cursor(
                self.twitter.search, q=query,
                tweet_mode="extended"
            )

            import pdb
            pdb.set_trace()
            for status_quoting in cursor_quote.items():
                statuses.append(status_quoting)

        for s in statuses:
            s_author_name = s.author.screen_name
            s_id = s.id

            # tree object storing comment thread
            thread = treelib.Tree()
            thread.create_node(
                tag=hash(s_author_name), identifier=s_id, data=content
            )

            # cursor over replies to tweet
            replies = tweepy.Cursor(
                self.twitter.search,
                q=f"to:{s_author_name}",
                since_id=s_id,
                tweet_mode="extended",
            ).items()

            for i in range(limit):
                try:
                    reply = replies.next()

                    if (
                        reply.in_reply_to_status_id is not None
                        and reply.in_reply_to_status_id == s_id
                    ):

                        # obtain thread originated from current reply
                        subthread = self.__reply_to_thread__(reply, limit)

                        # add subthread as children of the current node
                        thread.paste(s_id, subthread)

                except tweepy.RateLimitError:
                    print("Twitter api rate limit reached")
                    time.sleep(60)
                    continue

                except StopIteration:
                    break

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
                status, keyword, limit, cross)
            threads = itertools.chain(threads, content_threads)

        return threads
