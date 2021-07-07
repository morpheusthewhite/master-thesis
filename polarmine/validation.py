from typing import List, Union
import numpy as np
from sklearn import metrics

from polarmine.graph import InteractionGraph
from polarmine.ecp.ecp_solver import ECPSolver
from polarmine.ecp.alternative.alternative_solver import AlternativeSolver


def clustering_accuracy(
    graph: InteractionGraph,
    vertices_assignment: np.ndarray,
    n_clusters: int,
    alpha: float,
    solver: Union[ECPSolver, AlternativeSolver],
) -> tuple[float, float, float, List[float], List[List[int]]]:
    current_edge_filter, _ = graph.graph.get_edge_filter()
    if current_edge_filter is None:
        current_edge_filter = graph.graph.new_edge_property("bool")
        current_edge_filter.a = np.ones_like(current_edge_filter.a)

    vertices_assignment = np.array(vertices_assignment)

    # array containing prediction of group for each vertex
    vertices_predicted: np.ndarray = np.empty_like(vertices_assignment)
    vertices_predicted[:] = -1

    iterations_score = []
    iterations_vertices = []

    for _ in range(n_clusters):
        if isinstance(solver, ECPSolver):
            _, vertices, _, nc_threads = solver.solve(graph, alpha)
        else:
            _, vertices = solver.solve(graph, alpha)
            nc_threads = []

        # TODO: find largest component?
        #  vertices = graph.largest_component_vertices(vertices)

        if len(vertices) == 0:
            break

        # compute jaccard coefficient for the current classification
        subgraph_vertices_assignment = vertices_assignment[vertices]
        majority_class = np.bincount(subgraph_vertices_assignment).argmax()

        # filter out vertices which were already predicted as part of an
        # echo chamber
        vertex_already_predicted = vertices_predicted > -1

        # assign a label only to vertices which weren't part of another
        # echo chamber
        current_vertex_prediction: np.ndarray = np.zeros_like(
            vertices_predicted
        )
        current_vertex_prediction[vertices] = majority_class + 1
        current_vertex_prediction = current_vertex_prediction * (
            1 - vertex_already_predicted
        )

        vertices_predicted += current_vertex_prediction

        induced_edges_property = graph.is_induced_edge(
            set(vertices), set(nc_threads)
        )

        # exclude induced vertices, i.e. keep edges that are not induced
        # and unfilter previously
        current_edge_filter.a = np.logical_and(
            current_edge_filter.a,
            np.logical_not(induced_edges_property.a),
        )
        graph.graph.set_edge_filter(current_edge_filter)

        class_assignment = (vertices_assignment == majority_class).astype(
            np.int32
        )
        class_prediction: np.ndarray = np.zeros_like(class_assignment)
        class_prediction[vertices] = 1
        iteration_score = metrics.jaccard_score(
            class_assignment, class_prediction
        )
        iterations_score.append(iteration_score)

        iterations_vertices.append(vertices)

    current_vertex_filter, _ = graph.graph.get_vertex_filter()
    selected_vertices = list(np.where(current_vertex_filter.a != 0)[0])

    # ignore the unselected nodes for computing the clustering statistics
    vertices_assignment = vertices_assignment[selected_vertices]
    vertices_predicted = vertices_predicted[selected_vertices]

    graph.clear_filters()

    adjusted_rand_score = metrics.adjusted_rand_score(
        vertices_assignment, vertices_predicted
    )
    rand_score = metrics.rand_score(vertices_assignment, vertices_predicted)

    jaccard_score: float = metrics.jaccard_score(
        vertices_assignment, vertices_predicted, average="micro"
    )

    return (
        adjusted_rand_score,
        rand_score,
        jaccard_score,
        iterations_score,
        iterations_vertices,
    )
