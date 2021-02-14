import requests
import logging
import sys
import time
from urllib.parse import urlencode

from tweepy.error import (
    is_rate_limit_error_message,
    RateLimitError,
    TweepError,
)
from tweepy.api import *
from tweepy.models import Model
from polarmine.tweepy.parsers_v2 import Parserv2

log = logging.getLogger(__name__)


class APIv2(API):
    def __init__(self, *args, **kwargs):
        API.__init__(self, *args, **kwargs)
        self.parserv2 = Parserv2()

    def requestv2(
        self,
        method,
        endpoint,
        *args,
        endpoint_parameters=(),
        params=None,
        headers=None,
        json_payload=None,
        parser=None,
        payload_list=False,
        payload_type=None,
        post_data=None,
        require_auth=True,
        return_cursors=False,
        upload_api=False,
        use_cache=True,
        **kwargs,
    ):
        # If authentication is required and no credentials
        # are provided, throw an error.
        if require_auth and not self.auth:
            raise TweepError("Authentication required!")

        self.cached_result = False

        # Build the request URL
        path = f"/2/{endpoint}"
        if upload_api:
            url = "https://" + self.upload_host + path
        else:
            url = "https://" + self.host + path

        if params is None:
            params = {}

        for idx, arg in enumerate(args):
            if arg is None:
                continue
            try:
                params[endpoint_parameters[idx]] = str(arg)
            except IndexError:
                raise TweepError("Too many parameters supplied!")

        for k, arg in kwargs.items():
            if arg is None:
                continue

            if k in params:
                raise TweepError(
                    f"Multiple values for parameter {k} supplied!"
                )
            if k not in endpoint_parameters:
                log.warning(f"Unexpected parameter: {k}")
            params[k] = str(arg)

        if params.get("tweet.fields") is None:
            tweet_fields = "id,referenced_tweets"
            params["tweet.fields"] = tweet_fields

        log.info("PARAMS: %r", params)

        # Query the cache if one is available
        # and this request uses a GET method.
        if use_cache and self.cache and method == "GET":
            cache_result = self.cache.get(f"{path}?{urlencode(params)}")

            # if cache result found and not expired, return it
            if cache_result:

                # must restore api reference
                if isinstance(cache_result, list):
                    for result in cache_result:
                        if isinstance(result, Model):
                            result._api = self
                        else:
                            if isinstance(cache_result, Model):
                                cache_result._api = self
                                self.cached_result = True
                                return cache_result

        # Monitoring rate limits
        remaining_calls = None
        reset_time = None

        if parser is None:
            parser = self.parser

        try:
            # Continue attempting request until successful
            # or maximum number of retries is reached.
            retries_performed = 0
            while retries_performed <= self.retry_count:
                if (
                    self.wait_on_rate_limit
                    and reset_time is not None
                    and remaining_calls is not None
                    and remaining_calls < 1
                ):
                    # Handle running out of API calls
                    sleep_time = reset_time - int(time.time())
                    if sleep_time > 0:
                        log.warning(
                            f"Rate limit reached. Sleeping for: {sleep_time}"
                        )
                        time.sleep(sleep_time + 1)  # Sleep for extra sec

                # Apply authentication
                auth = None
                if self.auth:
                    auth = self.auth.apply_auth()

                # Execute request
                try:
                    resp = self.session.request(
                        method,
                        url,
                        params=params,
                        headers=headers,
                        data=post_data,
                        json=json_payload,
                        timeout=self.timeout,
                        auth=auth,
                        proxies=self.proxy,
                    )
                except Exception as e:
                    raise TweepError(
                        f"Failed to send request: {e}"
                    ).with_traceback(sys.exc_info()[2])

                if 200 <= resp.status_code < 300:
                    break

                rem_calls = resp.headers.get("x-rate-limit-remaining")
                if rem_calls is not None:
                    remaining_calls = int(rem_calls)
                elif remaining_calls is not None:
                    remaining_calls -= 1

                reset_time = resp.headers.get("x-rate-limit-reset")
                if reset_time is not None:
                    reset_time = int(reset_time)

                retry_delay = self.retry_delay
                if resp.status_code in (420, 429) and self.wait_on_rate_limit:
                    if remaining_calls == 0:
                        # If ran out of calls before waiting switching retry last call
                        continue
                    if "retry-after" in resp.headers:
                        retry_delay = float(resp.headers["retry-after"])
                    elif (
                        self.retry_errors
                        and resp.status_code not in self.retry_errors
                    ):
                        # Exit request loop if non-retry error code
                        break

                # Sleep before retrying request again
                time.sleep(retry_delay)
                retries_performed += 1

            # If an error was returned, throw an exception
            self.last_response = resp
            if resp.status_code and not 200 <= resp.status_code < 300:
                try:
                    error_msg, api_error_code = parser.parse_error(resp.text)
                except Exception:
                    error_msg = f"Twitter error response: status code = {resp.status_code}"
                    api_error_code = None

                if is_rate_limit_error_message(error_msg):
                    raise RateLimitError(error_msg, resp)
                else:
                    raise TweepError(error_msg, resp, api_code=api_error_code)

            # Parse the response payload
            return_cursors = (
                return_cursors or "cursor" in params or "next" in params
            )
            result, next_token = self.parserv2.parse(
                resp.text,
                api=self,
                payload_list=payload_list,
                payload_type=payload_type,
                return_cursors=return_cursors,
            )

            # Store result into cache if one is available.
            if use_cache and self.cache and method == "GET" and result:
                self.cache.store(f"{path}?{urlencode(params)}", result)

            return result, next_token
        finally:
            self.session.close()

    @pagination(mode="id")
    @payload("search_results")
    def searchv2(self, *args, **kwargs):
        """:reference: https://developer.twitter.com/en/docs/twitter-api/tweets/search/migrate/standard-to-twitter-api-v2"""
        return self.requestv2(
            "GET",
            "tweets/search/recent",
            *args,
            endpoint_parameters=(
                "query",
                "tweet.fields",
                "user.fields",
                "lang",
                "locale",
                "since_id",
                "geocode",
                "until_id",
                "end_time",
                "start_time",
                "next_token",
                "result_type",
                "max_results",
                "include_entities",
                "expansions",
            ),
            **kwargs,
        )

    def get_replies_ids(self, conversation_id, status_author):
        # select only replies to author of the tweet
        query = f"conversation_id:{conversation_id} to:{status_author}"

        search_results, next_token = self.searchv2(
            query=query, max_results=100
        )

        # dictionary of replies, with key being the id of the parent and value
        # the id of the many replies
        replies = {}

        # iterate until a None next_token is received
        while next_token is not None:

            for reply in search_results:

                assert len(reply.referenced_tweets) == 1
                assert reply.referenced_tweets[0]["type"] == "replied_to"

                parent_status_id = reply.referenced_tweets[0]["id"]

                # if the key does not exist create a list with just this element
                # otherwise append it to the existing list
                if replies.get(parent_status_id) is None:
                    replies[parent_status_id] = [reply.id]
                else:
                    replies[parent_status_id].append(reply.id)

            search_results, next_token = self.searchv2(
                query=query, max_results=100, next_token=next_token
            )

        return replies
