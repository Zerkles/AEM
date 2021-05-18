from collections import Counter

from ..base import TimerOptimizer, Route, Solution, Optimizer
import numpy as np
import random
from typing import List, Set

from .. import LocalInnerEdgeOptimizer


def perturbations_type1(route: Route, n_verticies: int, n_iter: int):
    unused_vertices = set(list(range(0, n_verticies))) - set(route)

    for _ in range(n_iter):
        # random vertex replace
        vertex1 = random.choice(list(unused_vertices))
        vertex2 = random.randint(0, len(route))

        unused_vertices.add(route[vertex2])
        route[vertex2] = vertex1
        unused_vertices.remove(vertex1)

        # random edge replace
        edge_start = random.randint(0, len(route))
        edge_end = random.randint(0, len(route))
        while edge_end == edge_start:
            edge_end = random.randint(0, len(route))

        if edge_start < edge_end:
            e = edge_start
            edge_start = edge_end
            edge_end = e
        edge = route[edge_start:edge_end + 1]
        edge.reverse()

        new_route = route[0:edge_start] + edge + route[edge_end + 1:]
        route = new_route

    return route


class ILS1(TimerOptimizer):
    def _find_solution(self):
        best_solution = Solution(np.inf, self.route)
        route: Route = self.route[:]
        vertices = len(self.distance_matrix)
        opt = LocalInnerEdgeOptimizer(self.distance_matrix, route)
        solution = opt()

        while True:
            if solution.cost < best_solution.cost:
                best_solution = Solution(solution.cost, solution.route[:])

            route = perturbations_type1(solution.route, vertices, 2)
            opt = LocalInnerEdgeOptimizer(self.distance_matrix, route)
            solution = opt()

            yield best_solution


def greedy_cycle(distance_matrix: np.ndarray, route: List[int], unused_vertices: Set, to_restore: int) -> List[int]:
    end_len = len(route) + to_restore
    unused_vertices = list(unused_vertices)

    while len(route) != end_len:
        v1 = route[0]  # to na indeksach wszystko jest
        v2 = unused_vertices[0]
        v3 = route[1]
        dst = distance_matrix[v1][v2] + distance_matrix[v2][v3] - distance_matrix[v1][v3]
        best_move = (v1, v2, v3, dst)

        for i in range(len(route)):
            v1 = route[i - 1]
            v3 = route[i]

            for j in range(len(unused_vertices)):
                v2 = unused_vertices[j]
                dst = distance_matrix[v1][v2] + distance_matrix[v2][v3] - distance_matrix[v1][v3]

                if dst < best_move[2]:
                    best_move = (i, v2, dst)

        route.insert(best_move[0], best_move[1])
        unused_vertices.remove(best_move[1])

    return route


def perturbations_type2(route: Route, distance_matrix, n_verticies: int, percent: float) -> Route:
    unused_vertices = set(list(range(0, n_verticies))) - set(route)
    before_number = len(route)

    destroy_n_verticies = int(n_verticies * percent)
    for _ in range(destroy_n_verticies):
        vertex = random.choice(route)
        unused_vertices.add(vertex)
        route.remove(vertex)

    destroy_n_edges = int((len(route) / 2) * percent)
    for _ in range(destroy_n_edges):
        edge_end = random.randint(0, len(route) - 1)

        v = route[edge_end]
        v_prev = route[edge_end - 1]

        unused_vertices.add(v)
        unused_vertices.add(v_prev)

        route.remove(v)
        route.remove(v_prev)

    to_restore = before_number - len(route)
    route = greedy_cycle(distance_matrix, route, unused_vertices, to_restore)

    return Route(route)


class ILS2(TimerOptimizer):
    def _find_solution(self):
        route: Route = self.route[:]
        vertices = len(self.distance_matrix)
        best_solution = Solution(np.inf, self.route)
        opt = LocalInnerEdgeOptimizer(self.distance_matrix, route)
        optimal_solution = opt()
        solution = Solution(optimal_solution.cost, optimal_solution.route[:])

        while True:
            if solution.cost < best_solution.cost:
                best_solution = Solution(solution.cost, solution.route[:])

            route = perturbations_type2(optimal_solution.route, opt.distance_matrix, vertices, 0.07)
            cost = self.__calculate_cost(route)
            solution = Solution(cost, route[:])

            yield best_solution

    def __calculate_cost(self, route):
        return sum(
            self.distance_matrix[a, b]
            if a is not None and b is not None else None
            for a, b in zip(route, route[1:] + [route[0]])
        )
