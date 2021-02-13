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
    assert isinstance(search_result[0].id, str)


def test_complete_search():
    # retrieve a status and the full conversation related to it
    thread = list(
        twitter_collector.collect(
            1, page="EU_Commission", limit=1, cross=False
        )
    )[0]
    root_status = thread.nodes[thread.root]

    query = f"conversation_id:{root_status.identifier}"
    search_result, next_token = twitter_collector.twitter.searchv2(query=query)

    while next_token is not None:
        search_result, next_token = twitter_collector.twitter.searchv2(
            query=query, next_token=next_token
        )
        assert next_token is not None
        assert isinstance(search_result, SearchResultsv2)
        assert len(search_result) > 0

        # first status in the current page of results
        status_0 = search_result[0]
        assert isinstance(status_0.id, str)
