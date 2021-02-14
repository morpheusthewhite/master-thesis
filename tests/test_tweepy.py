from polarmine.tweepy.models_v2 import SearchResultsv2
from polarmine.collectors.twitter_collector import TwitterCollector


twitter_collector = TwitterCollector()


def test_simple_search():
    search_result, next_token = twitter_collector.twitter.searchv2(
        query="snow"
    )

    assert next_token is not None
    assert isinstance(next_token, str)
    assert isinstance(search_result, SearchResultsv2)


def test_long_search():
    search_result, next_token = twitter_collector.twitter.searchv2(
        query="snow"
    )
    assert next_token is not None

    search_result, next_token = twitter_collector.twitter.searchv2(
        query="snow", next_token=next_token
    )
    assert next_token is not None
    assert isinstance(search_result, SearchResultsv2)
    assert len(search_result) > 0
    assert isinstance(search_result[0].id, int)


def test_complete_search():
    # retrieve a status and the full conversation related to it
    author = "vonderleyen"
    statuses = list(
        twitter_collector.__find_statuses__(1, keyword=None, page=author)
    )
    root_status = statuses[0]

    query = f"conversation_id:{root_status.id} to:{author}"
    search_result, next_token = twitter_collector.twitter.searchv2(query=query)

    while next_token is not None:
        search_result, next_token = twitter_collector.twitter.searchv2(
            query=query, next_token=next_token
        )
        assert isinstance(search_result, SearchResultsv2)
        assert len(search_result) > 0

        # first status in the current page of results
        status_0 = search_result[0]
        assert isinstance(status_0.id, int)
        assert status_0.referenced_tweets is not None


def test_get_ids():
    # retrieve a status and the full conversation related to it
    author = "vonderleyen"
    statuses = list(
        twitter_collector.__find_statuses__(1, keyword=None, page=author)
    )
    root_status = statuses[0]

    replies_dict = twitter_collector.twitter.get_replies_id(
        root_status.id, author
    )

    for k, v in replies_dict.items():
        assert isinstance(k, int)
        assert isinstance(v, list)
        assert isinstance(v[0], int)
