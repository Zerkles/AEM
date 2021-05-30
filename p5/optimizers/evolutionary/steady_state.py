from typing import List, Tuple
from itertools import combinations

from ..base import TimerOptimizer, Route, Solution
from .. import LocalInnerEdgeOptimizer, RandomOptimizer
from random import sample
import numpy as np


class EvoRandomSearchOptimizer(TimerOptimizer):
    MUTATION_PROBA = 0.1

    @staticmethod
    def __parents_selection(population: List[Solution]) -> Tuple[Route, Route]:
        parent_a, parent_b = sorted(population, key=lambda x: x.cost)[:2]
        return parent_a.route, parent_b.route

    def __cross_features(self, parent_a: Route, parent_b: Route) -> Route:
        sequences_a = [[*parent_a[v_id:], *parent_a[:v_id]] for v_id in range(len(parent_a))]
        sequences_b = [[*parent_b[v_id:], *parent_b[:v_id]] for v_id in range(len(parent_b))]

        common = {}
        sequence_a_indices = range(len(parent_a))
        for loop_seq_a in sequences_a:
            for start_a, stop_a in combinations(sequence_a_indices, 2):
                sequence_a: Route = Route(loop_seq_a[start_a:stop_a])
                score_sequence_a = len(sequence_a)/self._calculate_score(sequence_a) # im wiecej wierz. tym lepiej
                seq_range = (sequence_a[0], sequence_a[-1])
                common[seq_range] = {'seq': sequence_a, 'score': score_sequence_a}

        sequence_b_indices = range(len(parent_b))
        for loop_seq_b in sequences_b:
            for start_b, stop_b in combinations(sequence_b_indices, 2):
                sequence_b: Route = Route(loop_seq_b[start_b:stop_b])
                score_sequence_b = len(sequence_b)/self._calculate_score(sequence_b) # im wiecej wierz. tym lepiej

                seq_range = (sequence_b[0], sequence_b[-1])
                if (seq_range in common and score_sequence_b < common[seq_range]['score']) or (seq_range not in common):
                    common[seq_range] = {'seq': sequence_b, 'score': score_sequence_b}

        sorted_keys = sorted(common, key=lambda k: common[k]['score'], reverse=True)
        child = Route([])
        for key in sorted_keys:
            new_seq = common[key]['seq']
            if not any(el in child for el in new_seq):
                child.extend(new_seq)

        return child

    def __mutation_features(self, parent_a: Route, parent_b: Route, child: Route):
        new_route = child
        indices = {*parent_a, *parent_b} - {*child}
        indices_amount = len(parent_a) - len(child)
        new_route.extend(sample(indices, indices_amount))

        unused = {*range(len(self.distance_matrix))} - {*new_route}

        # dodatkowa mutacja genów
        for i in range(len(new_route)):
            mutate = np.random.choice([False, True], p=[
                1 - EvoRandomSearchOptimizer.MUTATION_PROBA,
                EvoRandomSearchOptimizer.MUTATION_PROBA
            ])

            if mutate:
                new_gene = np.random.choice([*unused])
                unused.remove(new_gene)
                unused.add(new_route[i])
                new_route[i] = new_gene

        return new_route

    def _find_solution(self):
        initial_population_size = 20
        population = self._generate_population(initial_population_size)

        while 1:
            parents = self.__parents_selection(population)
            child_incomplete_route = self.__cross_features(*parents)
            child_route = self.__mutation_features(*parents, child_incomplete_route)
            child_route_cost = self._calculate_score(child_route)

            if child_route_cost < population[-1].cost:
                for i in reversed(range(initial_population_size)):
                    if child_route_cost > population[i].cost:
                        population.insert(i, Solution(child_route_cost, child_route))
                        population.pop()

            yield population[0]

    def _generate_population(self, initial_population_size):
        population = []

        for _ in range(initial_population_size):
            route: Route = self.route[:]
            opt = RandomOptimizer(self.distance_matrix, route, 1)
            solution = opt()

            population.append(solution)

        return population
