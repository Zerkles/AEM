from collections import defaultdict
from statistics import mean
from typing import List, Dict, Tuple

from ..inner_edge_optimizer import LocalInnerEdgeOptimizer
from ..base import TimerOptimizer, Route, Solution, Optimizer
import numpy as np
import random

from ...loader import Matrix2D


class GlobalConvexityOptimizer(Optimizer):
    def __init__(self, distance_matrix, points):
        super().__init__(distance_matrix, points)
        self.n_iter = 100

    def _get_edges(self, solution):
        edges = set()
        for v1 in solution.route:
            for v2 in solution.route[1:] + [solution.route[0]]:
                edges.add((v1, v2))
        return edges

    def _compare_solutions(self, solutions_sets, best_solution_set):
        results = []
        for s1 in solutions_sets:
            sims = []
            for s2 in solutions_sets:
                sims.append(len(s1[1].intersection(s2[1])))
            results.append((s1[0], len(s1[1].intersection(best_solution_set)), mean(sims)))

        return results

    def _calc_similarities_vertex(self, solutions: List[Solution], best_solution: Solution):
        solutions_sets = []
        best_solution_set = set(best_solution.route)

        for s in solutions:
            solutions_sets.append((s.cost, set(s.route)))

        return self._compare_solutions(solutions_sets, best_solution_set)

    def _calc_similarities_edge(self, solutions: List[Solution], best_solution: Solution):
        solutions_sets = []
        best_solution_set = self._get_edges(best_solution)

        for s in solutions:
            solutions_sets.append((s.cost, self._get_edges(s)))

        return self._compare_solutions(solutions_sets, best_solution_set)

    def _search(self):
        best_solution = Solution(np.inf, self.route)
        vertices = self.distance_matrix.shape[0]
        route: Route = Route([*range(vertices // 2)])

        solutions = [LocalInnerEdgeOptimizer(self.distance_matrix, route)()]

        for _ in range(self.n_iter):
            route: Route = Route([*range(vertices // 2)])
            opt = LocalInnerEdgeOptimizer(self.distance_matrix, route)
            sol = opt()

            solutions.append(sol)

            if sol.cost < best_solution.cost:
                best_solution = Solution(sol.cost, sol.route[:])

        solutions.remove(best_solution)

        v_sim = self._calc_similarities_vertex(solutions, best_solution)
        v_sim.sort(key=lambda x: x[0])

        e_sim = self._calc_similarities_edge(solutions, best_solution)
        e_sim.sort(key=lambda x: x[0])

        return v_sim, e_sim
