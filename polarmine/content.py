class Content():
    def __init__(self, url: str, text: str, time: float, author: str,
                 keyword: str = None, **kwargs):
        super(Content, self).__init__(**kwargs)

        self.url = url
        self.text = text
        self.time = time
        self.author = author
        self.keyword = keyword

    def __str__(self):
        return str(self.__dict__)
