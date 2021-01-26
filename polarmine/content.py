class Content():
    def __init__(self, url, text, date, author, keyword=None, **kwargs):
        super(Content, self).__init__(**kwargs)

        self.url = url
        self.text = text
        self.date = date
        self.author = author
        self.keyword = keyword

    def __str__(self):
        return str(self.__dict__)
