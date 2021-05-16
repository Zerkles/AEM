from abc import abstractmethod, ABC

from .base import Route, Solution, InnerOuterVertexOptimizer


class GlobalNeighborOptimizer(InnerOuterVertexOptimizer, ABC):
    def _search(self) -> Solution:
        best_solution = Solution(self.init_cost, self.route)

        while 1:
            solutions = [
                self._find_best_solution(best_solution.route),
                min(self._find_swap_inner_outer_vertices_solutions(best_solution.route), key=lambda x: x.cost)
            ]

            solution = min(solutions, key=lambda x: x.cost)
            cost = self._calculate_score(solution.route)

            if cost < best_solution.cost:
                best_solution = Solution(cost, solution.route)
            else:
                break

        return best_solution

    @abstractmethod
    def _find_best_solution(self, route: Route) -> Solution:
        pass
