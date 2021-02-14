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
            setattr(status, k, v)
        return status
