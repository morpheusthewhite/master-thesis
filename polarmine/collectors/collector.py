from abc import ABC, abstractmethod
from polarmine.content import Content


class Collector(ABC):
    """object able to collect content and the comment thread"""

    @abstractmethod
    def collect(
        self,
        ncontents,
        keyword: str = None,
        page: str = None,
        limit: int = 10000,
        cross: bool = True,
    ) -> list[Content]:
        """collect comment threads relative to content

        Args:
            ncontents: number of contents to be collected
            keyword: a keyword used for filtering content. If None, does not
            use any filter
            page: a starting page to retrieve the content. Note: must be
            compatible with the Collector itself
            limit: integer to limit band usage
            cross: include also repost of a certain content

        Returns:
            list[Content]: the list of retrieved contents with comments thread
        """
        pass
