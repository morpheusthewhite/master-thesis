import graph_tool.all as gt
import treelib


class PolarizationGraph(gt.Graph):

    """A graph class providing methods for polarization analysis """

    def __init__(self, threads: list[treelib.Tree], *args, **kwargs):
        super(gt.Graph, self).__init__(*args, **kwargs)

        # dictionary storing user:vertex_index
        self.users = {}

        for thread in threads:
            root_id = thread.root
            root = thread.nodes[root_id]

            # get the content, associated to the root node
            content = root.data
            # TODO: add content to edge as well as time, use content object

            # TODO: do you want to add the root to the graph? Seems so

            # iterate over all other nodes
            # initially the queue will contain just the root node children
            queue = [root]

            while len(queue) > 0:
                # remove one element from the queue
                node = queue.pop(0)
                node_identifier = node.identifier

                # get/create the corresponding vertex
                node_author = node.data.author
                node_vertex = self.get_user_vertex(node_author)

                # children of the current node
                children = thread.children(node_identifier)

                for child in children:
                    comment = child.data
                    comment_author = comment.author

                    # find the node if it is in the graph 
                    comment_vertex = self.get_user_vertex(comment_author)

                    # and add the edge
                    self.add_edge(comment_vertex, node_vertex)

                    # equeue this child
                    queue.append(child)

    def get_user_vertex(self, user):
        vertex_index = self.users.get(user)

        if vertex_index is None:
            vertex = self.add_vertex()

            # get the index and add it to the dictionary
            vertex_index = self.vertex_index[vertex]
            self.users[user] = vertex_index
        else:
            # retrieve the vertex object from the graph
            vertex = self.vertex(vertex_index)

        return vertex

