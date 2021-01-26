class Comment(object):

    """Object containing all comment information"""

    def __init__(self, text, author, time, **kwargs):
        super(Comment, self).__init__(**kwargs)

        self.author = author
        self.text = text
        self.time = time
