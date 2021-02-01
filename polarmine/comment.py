class Comment(object):

    """Object containing all comment information"""

    def __init__(self, text: str, author: str, time: float, **kwargs):
        super(Comment, self).__init__(**kwargs)

        self.author = author
        self.text = text
        self.time = time
