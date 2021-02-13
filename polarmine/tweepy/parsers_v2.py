from tweepy.parsers import JSONParser
from polarmine.tweepy.models_v2 import SearchResultsv2


class Parserv2(JSONParser):
    def parse(
        self,
        payload,
        *,
        api=None,
        payload_list=False,
        payload_type=None,
        return_cursors=False
    ):
        if payload_type is None:
            return

        json = JSONParser.parse(self, payload, return_cursors=return_cursors)
        if isinstance(json, tuple):
            json, cursors = json
        else:
            cursors = None

        if payload_list:
            result = SearchResultsv2.parse_list(api, json)
        else:
            result = SearchResultsv2.parse(api, json)

        if cursors:
            return result, cursors
        else:
            return result, json["meta"].get("next_token")
