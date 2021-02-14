from tweepy.models import ResultSet, Status


class SearchResultsv2(ResultSet):
    @classmethod
    def parse(cls, api, json):
        metadata = json["meta"]
        results = SearchResultsv2()
        results.count = metadata.get("result_count")

        if json.get("data") is not None:
            for status in json["data"]:
                results.append(Statusv2.parse(api, status))
        return results


class Statusv2(Status):
    @classmethod
    def parse(cls, api, json):
        status = cls(api)
        setattr(status, "_json", json)
        for k, v in json.items():
            if k == "id":

                # map id of tweet to int
                setattr(status, k, int(v))

            elif k == "referenced_tweets":

                # map id of referenced tweets to int
                for referenced_tweet in v:
                    referenced_tweet["id"] = int(referenced_tweet["id"])
                setattr(status, k, v)

            else:
                setattr(status, k, v)

        return status
