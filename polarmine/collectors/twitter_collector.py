import os
import math
import treelib
import tweepy
import itertools
import urllib
import validators
from typing import Optional, Iterator

from polarmine.collectors.collector import Collector
from polarmine.thread import Thread
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
        self.twitter = APIv2(auth, wait_on_rate_limit=True, retry_delay=10)

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
        nstatuses_found = 0
        statuses_found = []
        max_id = None

        while nstatuses_found < ncontents:
            nstatuses_remaining = ncontents - nstatuses_found

            if keyword is not None:
                cursor = tweepy.Cursor(
                    self.twitter.search,
                    q=f"{keyword} min_replies:{TWEET_MIN_REPLIES} -filter:replies",
                    tweet_mode="extended",
                    max_id=max_id,
                )
            elif page is not None:
                cursor = tweepy.Cursor(
                    self.twitter.user_timeline,
                    screen_name=page,
                    tweet_mode="extended",
                    exclude_replies=True,
                    max_id=max_id,
                )
            else:
                raise NotImplementedError

            # get the remaining number of statuses
            statuses_last = list(cursor.items(nstatuses_remaining))
            statuses_found.extend(statuses_last)

            if len(statuses_last) > 0:
                # use the id of the last (least recent) status retrieved
                # it is also the last in the list as the statuses are order
                # from the most recent to the oldest
                status_last = statuses_last[-1]
                # need to subtract 1 since in the API it is specified that the
                # tweet id equal to max_id is considered
                max_id = status_last.id - 1

                nstatuses_found += len(statuses_last)

        return statuses_found

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
            except (UnicodeDecodeError, UnicodeEncodeError):
                # unicode error, there are some invalid ASCII character in the request
                return iter([])
            except urllib.error.URLError:
                # certificate errors, may happen when the url is quite strange
                return iter([])
            except ConnectionResetError:
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

    def __status_to_discussion_tree__(
        self,
        status: tweepy.Status,
        root_data: object = None,
        limit: int = 10000,
    ) -> treelib.Tree:
        """Retrieve discussion tree associated to a certain status

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
        discussion_tree = treelib.Tree()

        if root_data is not None:
            # use provided data
            discussion_tree.create_node(
                tag=hash(status_author_name),
                identifier=status_id,
                data=root_data,
            )
        else:
            # create comment object, associated to root node of this tree
            # the tag of the node is the author of the tweet
            # this branch is tipically taken by quote replies
            comment_text = status.full_text
            comment_author = hash(status.author.screen_name)
            comment_time = status.created_at.timestamp()
            comment = Comment(comment_text, comment_author, comment_time)
            discussion_tree.create_node(
                tag=comment.author, identifier=status_id, data=comment
            )

        # get dictionary of replies
        # where each key is the str(id) of a tweet and the value is the
        # list of id of the replies to it
        replies_dict = self.twitter.get_replies_id(
            status_id, status_author_name
        )

        # initially the queue will contain only the root node
        queue = [status_id]
        i = 0

        while len(queue) > 0 and i < limit:
            reply = queue.pop(0)

            # replies to the current reply
            reply_replies = replies_dict.get(reply, [])

            # require 100 tweets at a time
            for i in range(math.ceil(len(reply_replies) / 100)):

                # cycle to handle "Connection reset by peer"
                fetched = False
                while not fetched:
                    try:
                        # probably needs int instead of string
                        statuses_batch = self.twitter.statuses_lookup(
                            reply_replies[i * 100 : (i + 1) * 100],
                            tweet_mode="extended",
                        )
                        fetched = True
                    except tweepy.error.TweepError:
                        print("Connection problems")

                for s in statuses_batch:
                    # create comment object, associated to node
                    # the tag of the node is the author of the tweet
                    comment_id = s.id
                    comment_text = s.full_text
                    comment_author = hash(s.author.screen_name)
                    comment_time = s.created_at.timestamp()
                    comment = Comment(
                        comment_text, comment_author, comment_time
                    )

                    discussion_tree.create_node(
                        tag=comment_author,
                        identifier=comment_id,
                        data=comment,
                        parent=reply,
                    )

            queue.extend(reply_replies)
            i += 1

        return discussion_tree

    def __status_to_content__(self, status: tweepy.Status) -> str:
        """Find content associated with the status, which will correspond to
        the url it shared if present, otherwise to the status url

        Args:
            status (tweepy.Status): status from which content is extracted

        Returns:
            str: the url of the content
        """
        # url of the status, eventually used as content url if no valid url
        # is found at the end of the tweet itself
        status_url = f"https://twitter.com/user/status/{status.id}"

        # check if url is present in the provided status (it is supposed to be at the end)
        status_text_url = status.full_text.split()[-1]

        # check if it is a valid url
        if validators.url(status_text_url):
            try:
                url_redirected = urllib.request.urlopen(status_text_url).url

            except urllib.error.HTTPError:
                # generic error, happens when tweet has some images?
                return status_url
            except (UnicodeDecodeError, UnicodeEncodeError):
                # unicode error, there are some invalid ASCII character in the request
                return status_url
            except urllib.error.URLError:
                # certificate errors, may happen when the url is quite strange
                return status_url

            url_parsed = urllib.parse.urlparse(url_redirected)

            # remove query parameters from the url
            url_cleaned = urllib.parse.urljoin(url_redirected, url_parsed.path)

            return url_cleaned
        else:
            return status_url

    def __status_to_discussion_trees__(
        self,
        status: tweepy.Status,
        keyword: str,
        limit: int,
        cross: bool,
        exclude_share: set,
    ) -> list[treelib.Tree]:
        """Find threads of comments associated to a certain status

        Args:
            status (tweepy.Status): the status for which threads are looked for
            keyword (str): the keyword used to filter status
            limit (int): maximum number of tweets to check when looking
            for replies
            exclude_share (set(str)): a set of contents for which cross is
            not performed even if `cross` is True. If None it is ignored, otherwise
            it will be updated with the content if not present in the set

        Returns:
            list[treelib.Tree]: the discussions trees associated with the
            status. The root node,
            corresponding to a root status, is associated with a
            `Thread` object in the node `data` while the other nodes have
            a `Comment` object
        """
        status_author_name = status.author.screen_name
        status_id = status.id

        # retrieve content url, associated to threads
        content = self.__status_to_content__(status)

        thread_url = f"https://twitter.com/user/status/{status_id}"
        thread_text = status.full_text
        thread_time = status.created_at.timestamp()
        thread_author = hash(status_author_name)
        thread = Thread(
            thread_url,
            thread_text,
            thread_time,
            thread_author,
            content,
            keyword,
        )

        # thread/tree including only replies to original status
        thread = self.__status_to_discussion_tree__(status, thread, limit)

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
                # exclude quote replies which also reply to some tweet to
                # prevent having duplicates (which would be detected among the
                # normal replies of the root tweet). This is a rare case, yet
                # some fancy guys like doing it.  This is an extreme solution,
                # in fact it would suffice to check that the current tweet
                # replies to another tweet which has not beed already fetched
                # nor it will be
                if quote_reply.in_reply_to_status_id is None:
                    # quote replies can be handled as normal status since their
                    # text is the reply (without including the quote)
                    discussion_subtree = self.__status_to_discussion_tree__(
                        quote_reply, limit=limit
                    )

                    # add subthread as children of the root
                    thread.paste(status_id, discussion_subtree)

            if exclude_share is None or content not in exclude_share:
                if exclude_share is not None:
                    exclude_share.add(content)

                # tweets which share the same url (usually pointing to an
                # external site)
                for status_share in self.__status_to_shares__(status):
                    # skip the original author's tweet
                    if status_share.id == status_id:
                        continue

                    # create content object, associated to root node
                    thread_share_url = (
                        f"https://twitter.com/user/status/{status_share.id}"
                    )
                    thread_share_text = status_share.full_text
                    thread_share_time = status_share.created_at.timestamp()
                    thread_share_author = hash(status_share.author.screen_name)
                    thread_share = Thread(
                        thread_share_url,
                        thread_share_text,
                        thread_share_time,
                        thread_share_author,
                        content,
                        keyword,
                    )

                    discussion_tree_share = self.__status_to_discussion_tree__(
                        status_share, limit=limit, root_data=thread_share
                    )

                    yield discussion_tree_share

        yield thread

    def collect(
        self,
        ncontents: int,
        keyword: str = None,
        page: str = None,
        limit: int = 10000,
        cross: bool = True,
    ) -> list[treelib.Tree]:
        """collect content and their relative comment threads

        Args:
            ncontents: number of contents to find
            keyword (Optional[str]): keyword used for filtering content.
            If page is not None then it is ignored
            page (Optional[str]): the starting page from which content is
            found.
            limit (int): maximum number of tweets to check when looking
            for replies
            cross (bool): if True includes also the retweets of the found statuses
            in the result

        Returns:
            list[Tree]: a list of tree, each associated to a thread.
            The root node is associated to the discussion root and its `data`
            is a Thread object, while for the other nodes it is a `Comment`
        """
        statuses = self.__find_statuses__(ncontents, keyword, page)
        discussion_trees = iter([])

        # set used to track mined contents this is necessary because some
        # twitter accounts like @nytimes tweet many times the same link so
        # contents in this set will be prevented from mining again tweets
        # sharing the same content/url. Also, since the @nytimes and other
        # similar accounts (like @foxnews) add some query parameters to the
        # link, their tweet will not be mined twice. Notice that if a user will
        # tweet twice the same identical url without query parameters then it
        # will be mined more than once
        # because generators are used, this set will be populated only once
        # discussion_trees are generated, so you probably don't want to use it
        # elsewhere
        contents = set()

        for status in statuses:

            content_discussion_trees = self.__status_to_discussion_trees__(
                status, keyword, limit, cross, contents
            )
            discussion_trees = itertools.chain(
                discussion_trees, content_discussion_trees
            )

        return discussion_trees
