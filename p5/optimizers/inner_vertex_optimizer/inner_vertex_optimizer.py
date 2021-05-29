from abc import ABC
from typing import Generator, List
import numpy as np
from ..base import Route, Solution, Optimizer, GlobalNeighborOptimizer, LocalNeighborOptimizer


class InnerVertexOptimizer(Optimizer, ABC):
    def _generate_solutions(self, route: Route) -> Generator[Solution, None, None]:
        for index, point in enumerate(route):
            prev_point = route[index - 1]
            next_point = route[index + 1 if index + 1 < len(route) else 0]

            # A -> B -> C ... D -> E -> F
            A_B = self.distance_matrix[prev_point][point]
            B_C = self.distance_matrix[point][next_point]
            A_B_C = A_B + B_C

            for ap_index, another_point in enumerate(route[index + 1:], index + 1):
                prev_another_point = route[ap_index - 1]
                next_another_point = route[ap_index + 1 if ap_index + 1 < len(route) else 0]

                D_E = self.distance_matrix[prev_another_point][another_point]
                E_F = self.distance_matrix[another_point][next_another_point]
                D_E_F = D_E + E_F

                # A -> E -> C ... D -> B -> F
                A_E = self.distance_matrix[prev_point][another_point]
                E_C = self.distance_matrix[another_point][next_point]
                A_E_C = A_E + E_C

                D_B = self.distance_matrix[prev_another_point][point]
                B_F = self.distance_matrix[point][next_another_point]
                D_B_F = D_B + B_F

                A_E_C_D_B_F = A_E_C + D_B_F
                A_B_C_D_E_F = A_B_C + D_E_F

                if A_E_C_D_B_F < A_B_C_D_E_F and A_E_C_D_B_F:
                    new_route = route[:]
                    new_route[index], new_route[ap_index] = new_route[ap_index], new_route[index]
                    yield Solution(A_E_C_D_B_F, new_route)


class GlobalInnerVertexOptimizer(GlobalNeighborOptimizer, InnerVertexOptimizer):
    def _find_best_solution(self, route: Route) -> Solution:
        try:
            return min([*self._generate_solutions(route)], key=lambda x: x.cost)
        except ValueError:
            return Solution(np.inf, Route([]))


class LocalInnerVertexOptimizer(LocalNeighborOptimizer, InnerVertexOptimizer):
    def _find_solutions(self, route: Route) -> List[Solution]:
        route = self._randomize_starting_point(route)
        route = self._randomize_direction(route)
        solutions = [*self._generate_solutions(route)]
        if len(solutions) == 0:
            solutions.append(Solution(np.inf, Route([])))
        return solutions
