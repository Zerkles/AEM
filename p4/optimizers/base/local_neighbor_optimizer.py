from abc import abstractmethod, ABC
from collections import deque
import random
from typing import List

from .base import Route, Solution, InnerOuterVertexOptimizer


class LocalNeighborOptimizer(InnerOuterVertexOptimizer, ABC):
    def _search(self) -> Solution:
        best_solution = Solution(self.init_cost, self.route)
        new_solution = True

        while new_solution:
            solutions = [
                *self._find_solutions(best_solution.route),
                *self._find_swap_inner_outer_vertices_solutions(best_solution.route)
            ]

            random.shuffle(solutions)

            for solution in solutions:
                cost = self._calculate_score(solution.route)

                if cost < best_solution.cost:
                    best_solution = Solution(cost, solution.route)
                    new_solution = True
                    break
            else:
                new_solution = False

        return best_solution

    @abstractmethod
    def _find_solutions(self, route: Route) -> List[Solution]:
        pass

    @staticmethod
    def _randomize_starting_point(route: Route) -> Route:
        pos = random.randint(0, len(route))
        route = deque(route)
        route.rotate(-pos)
        return Route([*route])

    @staticmethod
    def _randomize_direction(route: Route) -> Route:
        return [*reversed(route)] if random.randint(0, 1) else route


