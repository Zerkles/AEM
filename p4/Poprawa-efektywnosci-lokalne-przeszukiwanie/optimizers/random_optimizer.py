from loader import Matrix2D
from .base import TimerOptimizer, Route, Solution, Optimizer
import numpy as np
import random


class RandomOptimizer(TimerOptimizer):
    def _find_solution(self):
        best_solution = Solution(np.inf, self.route)
        route: Route = self.route[:]

        while True:
            random.shuffle(route)

            score = self._calculate_score(route)
            if score < best_solution.cost:
                best_solution = Solution(score, route[:])

            yield best_solution


class MultiStartLocalSearchOptimizer(Optimizer):

    def __init__(self, distance_matrix: Matrix2D, route: Route):
        super().__init__(distance_matrix, route)
        self.n_iter = 100

    def _search(self) -> Solution:
        best_solution = Solution(np.inf, self.route)
        route: Route = self.route[:]

        for _ in range(self.n_iter):
            random.shuffle(route)

            score = self._calculate_score(route)
            if score < best_solution.cost:
                best_solution = Solution(score, route[:])

        return best_solution
