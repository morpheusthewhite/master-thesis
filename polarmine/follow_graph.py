import graph_tool.all as gt


class FollowGraph(object):
    """Graph handling follow relationships"""

    def __init__(self, follow_dict: dict, *args, **kwargs):
        super(FollowGraph, self).__init__(*args, **kwargs)
        self.graph = gt.Graph()

        self.users_id = self.graph.new_vertex_property("long")

        vertex_dict = {}

        # create a vertex associated to each user
        for user in list(follow_dict.keys()):
            vertex_user = self.graph.add_vertex()

            # store the vertex associated with the user in the dictionary
            vertex_dict[user] = vertex_user

            # save the id in a property map
            self.users_id[vertex_user] = user

        for user, friends in follow_dict.items():
            vertex_user = vertex_dict.get(user)

            for friend in friends:
                # check if the current friend is in the graph
                vertex_friend = vertex_dict.get(friend)

                # if it is present in the graph then add an edge from user to
                # friend
                if vertex_friend is not None:
                    self.graph.add_edge(vertex_user, vertex_friend)

    def communities(self):
        """Find communities in the follow graph"""
        state = gt.minimize_blockmodel_dl(self.graph)

        return zip(self.users_id.a, state.get_blocks().a)
