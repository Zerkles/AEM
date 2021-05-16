from dataclasses import dataclass
from typing import NewType, List
from loader import Matrix2D
from abc import ABC, abstractmethod
import numpy as np

Route = NewType('Route', List[int])


@dataclass
class Solution:
    cost: int
    route: Route


class Optimizer(ABC):
    def __init__(self, distance_matrix: Matrix2D, route: Route):
        self.distance_matrix = distance_matrix
        self.route = route
        self.init_cost = self._calculate_score(self.route)

    def __call__(self) -> Solution:
        return self._search()

    @abstractmethod
    def _search(self) -> Solution:
        pass

    def _calculate_score(self, route: Route):
        try:
            return sum(
                self.distance_matrix[a, b]
                if a is not None and b is not None else None
                for a, b in zip(route, route[1:] + [route[0]])
            )
        except IndexError:
            return np.inf


class InnerOuterVertexOptimizer(Optimizer, ABC):
    def _find_swap_inner_outer_vertices_solutions(self, route: Route) -> List[Solution]:
        solutions = []
        unused_vertices = self.__get_unused_points(route)
        for index_point, point in enumerate(route):
            prev_point = route[index_point - 1]
            next_point = route[index_point + 1 if index_point + 1 < len(route) else 0]

            distance, neighbor = self.__find_nearest(next_point, point, prev_point, unused_vertices)
            if neighbor:
                new_route = route[:]
                new_route[index_point] = neighbor

                solutions.append(Solution(distance, new_route))

        if len(solutions) == 0:
            solutions.append(Solution(np.inf, Route([])))
        return solutions

    def __get_unused_points(self, route: Route) -> List[int]:
        vertices = self.distance_matrix.shape[0]
        return [p for p in range(vertices) if p not in route]

    def __find_nearest(self, next_point: int, point: int, prev_point: int, unused_vertices: List[int]):
        # czy powinnismy wybierac najblizszy wzgledem obecnego punktu czy liczyc roznice i wybrac najlepszy?
        prev_to_current = self.distance_matrix[prev_point][point]
        current_to_next = self.distance_matrix[point][next_point]
        optimal_dist = prev_to_current + current_to_next

        nearest = {'distance': optimal_dist, 'neighbor': None}

        for unused in unused_vertices:
            prev_to_unused = self.distance_matrix[prev_point][unused]
            unused_to_next = self.distance_matrix[unused][next_point]
            dist = prev_to_unused + unused_to_next

            if dist < nearest['distance']:
                nearest = {'distance': dist, 'neighbor': unused}

        return nearest['distance'], nearest['neighbor']
