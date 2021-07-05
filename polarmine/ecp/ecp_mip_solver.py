from typing import List, Union
import pulp

from polarmine.ecp.ecp_solver import ECPSolver
from polarmine.graph import PolarizationGraph


class ECPMIPSolver(ECPSolver):
    def __init__(self, relaxation: bool = False, *args, **kwargs):
        super(ECPMIPSolver, self).__init__(*args, **kwargs)
        self.relaxation = relaxation

    def solve(
        self, graph: PolarizationGraph, alpha: float
    ) -> Union[
        tuple[float, List[int], List[int], List[str]],
        tuple[float, List[float], List[float], List[float]],
    ]:

        variables_cat = pulp.LpContinuous if self.relaxation else pulp.LpBinary
        variables_lb = 0 if self.relaxation else None
        variables_ub = 1 if self.relaxation else None
        variables_cat_thread = (
            pulp.LpContinuous if alpha <= 0.5 else pulp.LpBinary
        )

        controversial_contents = graph.controversial_contents(alpha)
        if len(controversial_contents) == 0:
            # no controversial content and so no edge to be considered
            return 0.0, [], [], []

        model = pulp.LpProblem("echo-chamber-score", pulp.LpMaximize)
        vertices_variables = [
            pulp.LpVariable(
                f"y_{index}",
                cat=variables_cat,
                lowBound=variables_lb,
                upBound=variables_ub,
            )
            for index in graph.graph.get_vertices()
        ]

        # thread: ([negative edges vars],[edges variables]) dictionary
        thread_edges_dict = {}
        # dictionary of edge variables incident to edges
        vertices_edge_variables = {}
        # objective function of the problem
        objective = 0

        # thread: thread_variable (z_k)
        thread_k_vars = {}
        # list of all the edge variables x_ij
        edge_variables = []

        # additional needed constraints require additional variables
        # if alpha > 0.5
        # thread: ([negative a_ij vars],[a_ij variables]) dictionary
        thread_aij_edges_dict = {}
        # list of all the edge variables a_ij
        aij_edge_variables = []

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
                edge_var = pulp.LpVariable(
                    f"x_{source}_{target}_{i}",
                    lowBound=0,
                    upBound=1,
                )
                edge_variables.append(edge_var)
                # create the variable associated to this thread if it does not exist
                z_k = thread_k_vars.get(thread)
                if z_k is None:
                    z_k = pulp.LpVariable(
                        f"z_{hash(thread)}",
                        lowBound=0,
                        upBound=1,
                        cat=variables_cat_thread,
                    )
                    thread_k_vars[thread] = z_k

                if weight > 0:
                    objective += edge_var
                else:
                    objective -= edge_var

                model += edge_var <= vertices_variables[source]
                model += edge_var <= vertices_variables[target]
                model += edge_var <= z_k
                model += (
                    edge_var
                    >= -2
                    + vertices_variables[source]
                    + vertices_variables[target]
                    + z_k
                )

                edges_negative_var, edges_var = thread_edges_dict.get(
                    thread, ([], [])
                )

                if weight < 0:
                    edges_negative_var.append(edge_var)
                edges_var.append(edge_var)

                thread_edges_dict[thread] = (edges_negative_var, edges_var)

                # add edge among edge variables of source
                source_edge_variables = vertices_edge_variables.get(source, [])
                source_edge_variables.append(edge_var)
                vertices_edge_variables[source] = source_edge_variables

                # add edge among edge variables of target
                target_edge_variables = vertices_edge_variables.get(target, [])
                target_edge_variables.append(edge_var)
                vertices_edge_variables[target] = target_edge_variables

                # additional constraints for alpha > 0.5

                if alpha > 0.5:
                    aij_edge_var = pulp.LpVariable(
                        f"a_{source}_{target}_{i}",
                        lowBound=0,
                        upBound=1,
                    )

                    model += (
                        aij_edge_var
                        >= -1
                        + vertices_variables[source]
                        + vertices_variables[target]
                    )
                    model += aij_edge_var <= vertices_variables[source]
                    model += aij_edge_var <= vertices_variables[target]

                    edges_negative_var, edges_var = thread_aij_edges_dict.get(
                        thread, ([], [])
                    )

                    if weight < 0:
                        edges_negative_var.append(aij_edge_var)
                    edges_var.append(aij_edge_var)

                    thread_aij_edges_dict[thread] = (
                        edges_negative_var,
                        edges_var,
                    )
                    aij_edge_variables.append(aij_edge_var)

        # add thread controversy constraints
        for _, edges_var_tuple in enumerate(thread_edges_dict.values()):
            edges_negative_var, edges_var = edges_var_tuple

            # sum of variables associated to negative edges of a single thread
            neg_edges_sum = pulp.lpSum(edges_negative_var)
            # sum of variables associated to edges of a single thread
            edges_sum = pulp.lpSum(edges_var)

            model += neg_edges_sum - alpha * edges_sum <= 0

        epsilon = 1e-30
        if alpha > 0.5:
            for thread, edges_var_tuple in thread_aij_edges_dict.items():
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
                model += (
                    neg_edges_sum - alpha * edges_sum + epsilon >= -N_k * z_k
                )

        # add constraint for setting to one only vertices where at least one
        # edge is at 1
        for (
            vertex_index,
            vertex_edge_variables,
        ) in vertices_edge_variables.items():
            model += vertices_variables[vertex_index] <= pulp.lpSum(
                vertex_edge_variables
            )

        model += objective
        model.solve()

        score = pulp.value(model.objective)

        if score == 0:
            return 0, [], [], []

        users: Union[List[int], List[float]] = []
        for i, vertex_variable in enumerate(vertices_variables):
            if self.relaxation:
                # if relaxation problem, return value of all the vertices
                # instead of indices of non-zero nodes
                users.append(pulp.value(vertex_variable))
            elif pulp.value(vertex_variable) == 1:
                users.append(i)

        edges: Union[List[int], List[float]] = []
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
            elif pulp.value(edge_variable) == 1:
                edge = (source, target, pulp.value(edge_variable))
                edges.append(edge)

        nc_threads: Union[List[str], List[float]] = []
        for thread, thread_variable in thread_k_vars.items():

            thread_value = pulp.value(thread_variable)
            if self.relaxation:
                # if relaxation problem, return value of all the vertices
                # instead of indices of non-zero threads
                nc_threads.append(thread_value)
            elif thread_value == 1:
                nc_threads.append(thread)

        return score, users, edges, nc_threads
