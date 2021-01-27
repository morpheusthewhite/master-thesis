import os
import pytest


from polarmine.collectors.reddit_collector import RedditCollector
from polarmine.collectors.twitter_collector import TwitterCollector


reddit_collector = RedditCollector()
twitter_collector = TwitterCollector()


def test_reddit_collect_simple():
    # simple check on single content
    contents = list(reddit_collector.collect(1))
    assert len(contents) == 1


def test_reddit_collect_more():
    # try to collect more than 1 content
    contents = list(reddit_collector.collect(2))
    assert len(contents) == 2


def test_reddit_collect_page():
    # try to collect from page
    contents = list(reddit_collector.collect(2, page="programming"))
    assert len(contents) == 2


def test_reddit_collect_keyword():
    # try to collect from keyword
    contents = list(reddit_collector.collect(2, keyword="obama"))
    assert len(contents) == 2
#
#  def test_twitter_init_wrong_env():
#      # instantiate a TwitterCollector with wrong environment
#      env_var = "TWITTER_CONSUMER_KEY"
#      env_value = os.getenv(env_var)
#
#
#      # remove the variable and later restore it
#      os.unsetenv(env_var)
#      print(os.getenv(env_var))
#
#      with pytest.raises(Exception):
#          TwitterCollector()
#
#      os.putenv(env_var, env_value)
