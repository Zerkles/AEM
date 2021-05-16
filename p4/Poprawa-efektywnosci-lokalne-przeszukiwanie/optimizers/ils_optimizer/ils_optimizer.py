from collections import Counter
from math import factorial

from loader import Matrix2D
from ..base import TimerOptimizer, Route, Solution, Optimizer
import numpy as np
import random
from typing import List

from .. import LocalInnerEdgeOptimizer


def perturbations_type1(route: Route, n_verticies: int, n_iter: int):
    unused_vertices = set(list(range(0, n_verticies))) - set(route)

    for _ in range(n_iter):

        # random replace
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

        while True:
            opt = LocalInnerEdgeOptimizer(self.distance_matrix, route)
            solution = opt()

            if solution.cost < best_solution.cost:
                best_solution = Solution(solution.cost, solution.route[:])
                route = perturbations_type1(solution.route, vertices, 2)

            yield best_solution


def nng(distance_matrix: np.ndarray, route, unused_vertices, to_restore: int) -> List[int]:
    nearest = []
    for p in route:
        n_dist, p_unused = min([distance_matrix[p, uv], uv] for uv in unused_vertices)
        nearest.append([n_dist, p, p_unused])

    print('NEAREST', len(nearest))
    used = 0
    for _, p, p_unused in sorted(nearest):
        if p_unused in unused_vertices:
            i = route.index(p)
            route.insert(i, p_unused)
            unused_vertices.remove(p_unused)
            used += 1

            if used == to_restore:
                print('REACHED')
                break
    else:
        print('Unused', Counter([p_unused for _, p, p_unused in nearest]))
        print('Unused vertices', unused_vertices)
        print('Po wyjsciu z petli ...', len(unused_vertices))

    return route


def perturbations_type2(route: Route, distance_matrix, n_verticies: int, percent: float):
    print('START', len(route))
    unused_vertices = set(list(range(0, n_verticies))) - set(route)
    before_number = len(route)

    destroy_n_verticies = int(n_verticies * percent)
    for _ in range(destroy_n_verticies):
        vertex = random.choice(route)
        unused_vertices.add(vertex)
        route.remove(vertex)

        route = nng(distance_matrix, route, unused_vertices, 1)

    destroy_n_edges = int((len(route) / 2) * percent)
    for _ in range(destroy_n_edges):
        edge_end = random.randint(0, len(route) - 1)

        unused_vertices.add(route[edge_end])
        unused_vertices.add(route[edge_end - 1])

        del route[edge_end]
        del route[edge_end - 1]

        route = nng(distance_matrix, route, unused_vertices, 1)

    return route


class ILS2(TimerOptimizer):
    def _find_solution(self):
        route: Route = self.route[:]
        vertices = len(self.distance_matrix)

        opt = LocalInnerEdgeOptimizer(self.distance_matrix, route)
        best_solution = opt()
        yield best_solution

        route = best_solution.route
        while True:
            route = perturbations_type2(route, opt.distance_matrix, vertices, 0.1)
            cost = sum(
                self.distance_matrix[a, b]
                if a is not None and b is not None else None
                for a, b in zip(route, route[1:] + [route[0]])
            )

            if cost < best_solution.cost:
                best_solution = Solution(cost, route[:])
                yield best_solution
