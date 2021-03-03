from treelib import Tree


class Thread:
    def __init__(
        self,
        url: str,
        text: str,
        time: float,
        author: int,
        content: str,
        keyword: str = None,
        **kwargs
    ):
        super(Thread, self).__init__(**kwargs)

        self.url = url
        self.text = text
        self.time = time
        self.author = author
        self.content = content
        self.keyword = keyword

    def __str__(self):
        return str(self.__dict__)
