from typing import List
import pulp

from polarmine.decp.decp_solver import DECPSolver
from polarmine.ecp.utils import score_from_vertices_index
from polarmine.graph import InteractionGraph


class DECPMIPSolver(DECPSolver):
    def __init__(self, relaxation: bool = False, *args, **kwargs):
        super(DECPMIPSolver, self).__init__(*args, **kwargs)
        self.relaxation = relaxation

    def solve(
        self, graph: InteractionGraph, alpha: float
    ) -> tuple[float, List[int], List[int], List[str]]:

        variables_cat = pulp.LpContinuous if self.relaxation else pulp.LpBinary
        variables_lb = 0 if self.relaxation else None
        variables_ub = 1 if self.relaxation else None

        controversial_contents = graph.controversial_contents(alpha)

        model = pulp.LpProblem("densest-echo-chamber-score", pulp.LpMaximize)
        vertices_binary_variables = [
            pulp.LpVariable(
                f"b_{index}",
                cat=variables_cat,
                lowBound=variables_lb,
                upBound=variables_ub,
            )
            for index in graph.graph.get_vertices()
        ]
        vertices_continous_variables = [
            pulp.LpVariable(
                f"y_{index}",
                lowBound=0,
            )
            for index in graph.graph.get_vertices()
        ]

        model += pulp.lpSum(vertices_continous_variables) == 1
        # each continous variable must activate the corresponding binary one
        for vertex_continous_variable, vertex_binary_variable in zip(
            vertices_continous_variables, vertices_binary_variables
        ):
            model += vertex_continous_variable <= vertex_binary_variable

            for vertex_continous_variable2 in vertices_continous_variables:
                if vertex_continous_variable != vertex_continous_variable2:
                    model += (
                        vertex_continous_variable
                        >= -1
                        + vertex_binary_variable
                        + vertex_continous_variable2
                    )

        # thread: ([negative edges vars],[edges variables]) dictionary
        thread_edges_dict = {}
        # objective function of the problem
        objective = 0

        # thread: thread_variable (z_k) dictionary
        thread_k_vars = {}
        # list of all the edge variables x_ij
        edge_variables = []

        for i, edge in enumerate(graph.graph.edges()):
            thread_obj = graph.threads[edge]
            content = thread_obj.content

            # ignore non controversial contents
            if content in controversial_contents:
                source, target = tuple(edge)
                source = int(source)
                target = int(target)
                weight = graph.weights[edge]

                thread = thread_obj.url
                edge_binary_var = pulp.LpVariable(
                    f"a_{source}_{target}_{i}",
                    cat=variables_cat,
                    lowBound=variables_lb,
                    upBound=variables_ub,
                )
                edge_continous_var = pulp.LpVariable(
                    f"x_{source}_{target}_{i}",
                    lowBound=0,
                )
                edge_variables.append(edge_continous_var)
                # create the variable associated to this thread if it does not exist
                z_k = thread_k_vars.get(thread)
                if z_k is None:
                    z_k = pulp.LpVariable(
                        f"z_{hash(thread)}",
                        lowBound=0,
                        upBound=1,
                        cat=variables_cat,
                    )
                    thread_k_vars[thread] = z_k

                if weight >= 0:
                    objective += edge_continous_var
                else:
                    objective -= edge_continous_var

                model += (
                    edge_continous_var <= vertices_continous_variables[source]
                )
                model += (
                    edge_continous_var <= vertices_continous_variables[target]
                )
                model += edge_continous_var <= edge_binary_var
                model += edge_continous_var <= z_k

                model += edge_binary_var <= vertices_binary_variables[source]
                model += edge_binary_var <= vertices_binary_variables[target]

                model += (
                    edge_binary_var
                    >= -1
                    + vertices_binary_variables[source]
                    + vertices_binary_variables[target]
                )

                model += (
                    edge_continous_var
                    >= -2
                    + edge_binary_var
                    + z_k
                    + vertices_continous_variables[source]
                )
                model += (
                    edge_continous_var
                    >= -2
                    + edge_binary_var
                    + z_k
                    + vertices_continous_variables[target]
                )

                edges_negative_var, edges_var = thread_edges_dict.get(
                    thread, ([], [])
                )

                if graph.weights[edge] < 0:
                    edges_negative_var.append(edge_binary_var)
                edges_var.append(edge_binary_var)

                thread_edges_dict[thread] = (edges_negative_var, edges_var)

        epsilon = 1e-30
        # add thread controversy constraints
        for thread, edges_var_tuple in thread_edges_dict.items():
            edges_negative_var, edges_var = edges_var_tuple
            z_k = thread_k_vars[thread]

            # sum of variables associated to negative edges of a single thread
            neg_edges_sum = pulp.lpSum(edges_negative_var)
            # sum of variables associated to edges of a single thread
            edges_sum = pulp.lpSum(edges_var)

            N_k = (len(edges_var) - len(edges_negative_var) + 1) * alpha
            M_k = (len(edges_negative_var) + 1) * (1 - alpha)

            model += neg_edges_sum - alpha * edges_sum + epsilon <= M_k * (
                1 - z_k
            )
            model += neg_edges_sum - alpha * edges_sum + epsilon >= -N_k * z_k

        model += objective
        model.solve()

        score = pulp.value(model.objective)

        users = []
        for i, vertex_variable in enumerate(vertices_continous_variables):
            if self.relaxation:
                # if relaxation problem, return value of all the vertices
                # instead of indices of non-zero nodes
                users.append(pulp.value(vertex_variable))
            elif pulp.value(vertex_variable) > 0:
                users.append(i)

        edges = []
        for i, edge_variable in enumerate(edge_variables):
            edge_name = edge_variable.name
            edge_name_split = edge_name.split("_")
            source = int(edge_name_split[1])
            target = int(edge_name_split[2])

            if self.relaxation:
                edge = (source, target, pulp.value(edge_variable))
                # if relaxation problem, return value of all the vertices
                # instead of indices of non-zero edges
                edges.append(edge)
            elif pulp.value(edge_variable) > 0:
                edge = (source, target, pulp.value(edge_variable))
                edges.append(edge)

        # Iterating over thread variables is not sufficient for detecting
        # active threads. In fact with this model formulation there can be some
        # variables set to 1 which do not actually contribute (this because
        # some vertices may have a 1 binary variable while still not taking any
        # of the available weight and so having the continuous variable set to
        # 0)
        #
        # nc_threads = []
        # for i, thread_variable in enumerate(thread_k_vars.values()):
        #
        #     thread_value = pulp.value(thread_variable)
        #     if relaxation:
        #         # if relaxation problem, return value of all the vertices
        #         # instead of indices of non-zero threads
        #         nc_threads.append(thread_value)
        #     elif thread_value == 1:
        #         nc_threads.append(i)

        _, nc_threads = score_from_vertices_index(graph, users, alpha)

        return score, users, edges, nc_threads
