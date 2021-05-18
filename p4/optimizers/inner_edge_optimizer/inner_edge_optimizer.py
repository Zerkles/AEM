from abc import ABC
from typing import Generator, List, Tuple
import numpy as np
from ..base import Route, Optimizer, Solution, GlobalNeighborOptimizer, LocalNeighborOptimizer


class InnerEdgeOptimizer(Optimizer, ABC):
    def _generate_solutions(self, route: Route) -> Generator[Solution, None, None]:
        for index, point in enumerate(route[:-3]):
            for ap_index, another_point in self.__find_k_nearest(index, point, route):
                rev = [*reversed(route[index + 1:ap_index])]

                # A -> B -> ... -> C -> D
                A_B = self.distance_matrix[point][rev[-1]]
                C_D = self.distance_matrix[rev[0]][another_point]
                A_B_C_D = A_B + C_D

                A_C = self.distance_matrix[point][rev[0]]
                B_D = self.distance_matrix[rev[-1]][another_point]
                A_C_B_D = A_C + B_D

                if A_C_B_D < A_B_C_D:
                    new_route = route[:index + 1] + rev + route[ap_index:]
                    yield Solution(A_C_B_D, new_route)

    def __find_k_nearest(self, index: int, point: int, route: Route) -> List[Tuple[int, int]]:
        neighbors = [(self.distance_matrix[point][another_point], ap_index, another_point)
                     for ap_index, another_point in enumerate(route[index + 3:], index + 3)]

        return [(ap_index, another_point) for _, ap_index, another_point in sorted(neighbors)][:5]


class GlobalInnerEdgeOptimizer(GlobalNeighborOptimizer, InnerEdgeOptimizer):
    def _find_best_solution(self, route: Route) -> Solution:
        try:
            return min([*self._generate_solutions(route)], key=lambda x: x.cost)
        except ValueError:
            return Solution(np.inf, Route([]))


class LocalInnerEdgeOptimizer(LocalNeighborOptimizer, InnerEdgeOptimizer):
    def _find_solutions(self, route: Route) -> List[Solution]:
        route = self._randomize_starting_point(route)
        route = self._randomize_direction(route)
        solutions = [*self._generate_solutions(route)]
        if len(solutions) == 0:
            solutions.append(Solution(np.inf, Route([])))
        return solutions
