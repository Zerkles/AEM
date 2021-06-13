from collections import deque, defaultdict
from typing import Generator

from loader import Matrix2D

from .inner_edge_optimizer import LocalInnerEdgeOptimizer
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
        self.n_iter = 20

    def _search(self) -> Solution:
        best_solution = Solution(np.inf, self.route)
        route: Route = self.route[:]

        for _ in range(self.n_iter):
            random.shuffle(route)
            opt = LocalInnerEdgeOptimizer(self.distance_matrix, route)
            sol = opt()

            if sol.cost < best_solution.cost:
                best_solution = Solution(sol.cost, sol.route[:])

        return best_solution


class AdaptiveMultiStartLocalSearchOptimizer(TimerOptimizer):
    def __init__(self, distance_matrix: Matrix2D, route: Route, time_ps: float):
        super().__init__(distance_matrix, route, time_ps)
        self.freq_table = defaultdict(int)

    def _find_solution(self) -> Generator[Solution, None, None]:
        best_solution = Solution(np.inf, self.route)
        route: Route = self.route[:]
        self.vertices = len(self.route)

        while 1:
            opt = LocalInnerEdgeOptimizer(self.distance_matrix, route)
            sol = opt()

            self.__update_freq_table(sol.route)
            route = self.__build_route(route)

            if sol.cost < best_solution.cost:
                best_solution = Solution(sol.cost, sol.route[:])

            yield best_solution

    def __build_route(self, init_route: Route) -> Route:
        random.shuffle(init_route)

        p = np.array([*self.freq_table.values()], dtype=np.float32)
        p /= np.sum(p)

        seq = np.random.choice(np.array([*self.freq_table.keys()], dtype=np.object), p=p)
        seq_len = len(seq)
        possible_inserts = list(range(self.vertices - seq_len))
        pos = np.random.choice(possible_inserts)

        route = [*init_route[:pos], *seq, *init_route[pos + seq_len:]]
        return Route(route)

    def __update_freq_table(self, route: Route):
        for seq_len in range(2, self.vertices - 1):
            for i in range(0, self.vertices, seq_len):
                sub_seq = route[i:i + seq_len]
                self.freq_table[tuple(sub_seq)] += 1
