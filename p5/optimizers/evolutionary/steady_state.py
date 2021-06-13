from typing import List, Tuple

from ..base import TimerOptimizer, Route, Solution
from .. import LocalInnerEdgeOptimizer, GlobalInnerEdgeOptimizer
import numpy as np
import random


class EvoRandomSearchOptimizer(TimerOptimizer):
    MUTATION_PROBA = 0.2

    @staticmethod
    def __parents_selection(population: List[Solution]) -> Tuple[Route, Route]:
        parent_a, parent_b = np.random.choice(population, 2)
        return parent_a.route, parent_b.route

    @staticmethod
    def __cross_features(parent_a: Route, parent_b: Route) -> Route:
        common_sub_seq = []
        for seq_len in range(1, min([len(parent_a), len(parent_b)]) + 1):
            x = {tuple(parent_a[start_seq_a:start_seq_a + seq_len]) for start_seq_a in range(0, len(parent_a), seq_len)}
            y = {tuple(parent_b[start_seq_b:start_seq_b + seq_len]) for start_seq_b in range(0, len(parent_b), seq_len)}

            common_sub_seq.extend(list(x & y))

        child = random.choice([parent_a, parent_b])[:]
        vertices = len(set(child))

        random.shuffle(common_sub_seq)
        for selected_seq in common_sub_seq:
            possible_inserts = list(range(len(child) - len(selected_seq)))
            random.shuffle(possible_inserts)

            for insert_pos in possible_inserts:
                new_child = [*child[:insert_pos], *selected_seq, *child[insert_pos + len(selected_seq):]]
                if vertices == len(set(new_child)):
                    return Route(new_child)
        else:
            raise ValueError('seq error')

    def __mutation(self, child: Route) -> Route:
        unused = {*range(len(self.distance_matrix))} - {*child}
        for p in range(len(child)):
            if np.random.uniform() > (1 - self.MUTATION_PROBA):
                v = np.random.choice(list(unused))
                unused.remove(v)
                unused.add(child[p])
                child[p] = v

        return child

    def _find_solution(self):
        initial_population_size = 20
        population = self._generate_population(initial_population_size)
        population.sort(key=lambda x: x.cost)

        while 1:
            parents = self.__parents_selection(population)
            child_route = self.__cross_features(*parents)
            child_route = self.__mutation(child_route)
            solution = LocalInnerEdgeOptimizer(self.distance_matrix, child_route)()

            if solution.cost < population[-1].cost:
                for i in range(initial_population_size):
                    if solution.cost < population[i].cost:
                        population.insert(i, solution)
                        population.pop()
                        break

            yield population[0]

    def _generate_population(self, initial_population_size):
        population = []

        for _ in range(initial_population_size):
            route: Route = self.route[:]
            np.random.shuffle(route)
            cost = self._calculate_score(route)
            population.append(Solution(cost, route))

        return population
