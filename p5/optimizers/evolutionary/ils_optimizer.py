from ..base import TimerOptimizer, Route, Solution
import numpy as np

from .. import LocalInnerEdgeOptimizer


class EvoOptimizer(TimerOptimizer):
    def _find_solution(self):
        initial_population_size = 20
        population = []

        for _ in range(initial_population_size):
            route: Route = self.route[:]
            opt = LocalInnerEdgeOptimizer(self.distance_matrix, route)
            solution = opt()

            population.append(solution)

        population.sort(key=lambda s: s.cost)

        while True:
            parents = population[:2]
            child = None  # TODO: generate new child

            if child.cost < population[-1].cost:
                for i in reversed(range(initial_population_size)):
                    if child.cost > population[i].cost:
                        population.insert(i, child)
                        population.pop()

            yield population[0]
