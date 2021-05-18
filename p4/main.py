from collections import defaultdict
from typing import List

import pandas as pd
from sys import argv
from loader import iterate_instances
from time import time
import random
import numpy as np
import matplotlib.pyplot as plt
import sys

from optimizers import Route, MultiStartLocalSearchOptimizer
from optimizers.base import Solution
from optimizers.ils_optimizer.ils_optimizer import ILS1, ILS2

pd.options.display.max_columns = None
pd.options.display.max_rows = None


def visualize_route(route: Route, points, path: str, filename: str):
    fig, ax = plt.subplots()

    x, y = [points[v_id][1] for v_id in route], [points[v_id][2] for v_id in route]

    x.append(x[0])
    y.append(y[0])

    ax.scatter(x, y)
    ax.plot(x, y)

    x, y = [v_id[1] for v_id in points if v_id not in route], [v_id[2] for v_id in points if v_id not in route]
    ax.scatter(x, y)
    fig.savefig(path + '/' + filename)


def main(instances_path: str, repeat: int, output_path: str):
    print('[STARTED]')

    optimizers = [
        ILS2  # TODO: Dodać ILS1
    ]

    for x in iterate_instances(instances_path):
        t, c = {}, {}
        best_solutions_routes = [Solution(np.inf, Route([])) for _ in range(3)]
        fname, distance_matrix, points = x.name, x.distance_matrix, x.points

        routes = []
        print('[INSTANCE]', fname)
        inform.write(f'[INSTANCE] {fname}\n')

        time_local = []
        tmp_t, tmp_c = [], []
        for i in range(repeat):
            inform.write(f'MSLS [PROGRESS {i + 1}/{repeat}]\n')
            vertices = distance_matrix.shape[0]
            route = Route([*range(vertices // 2)])
            random.shuffle(route)
            routes.append(route)

            begin = time()
            opt = MultiStartLocalSearchOptimizer(distance_matrix, route)
            solution = opt()

            end = time()
            time_local.append(end - begin)

            tmp_t.append(end - begin)
            tmp_c.append(solution.cost)

            if solution.cost < best_solutions_routes[0].cost:
                best_solutions_routes[0] = solution

        c[MultiStartLocalSearchOptimizer.__name__] = tmp_c
        t[MultiStartLocalSearchOptimizer.__name__] = tmp_t
        time_ps = np.mean(time_local)

        tmp_t, tmp_c = defaultdict(lambda: []), defaultdict(lambda: [])
        for i in range(repeat):
            inform.write(f'[PROGRESS {i + 1}/{repeat}]\n')
            vertices = distance_matrix.shape[0]
            route = Route([*range(vertices // 2)])
            random.shuffle(route)
            routes.append(route)

            for oid, Optimizer in enumerate(optimizers, 1):
                begin = time()
                opt = Optimizer(distance_matrix, route, 10)  # TODO: Ustawić czas na time_ps
                solution = opt()
                end = time()

                tmp_t[Optimizer.__name__].append(end - begin)
                tmp_c[Optimizer.__name__].append(solution.cost)

                if solution.cost < best_solutions_routes[oid].cost:
                    best_solutions_routes[oid] = solution

        c = {**c, **tmp_c}
        t = {**t, **tmp_t}

        max_len = max(len(v) for v in t.values())
        for k in t:
            t[k].extend([None for _ in range(max_len - len(t[k]))])

        max_len = max(len(v) for v in c.values())
        for k in c:
            c[k].extend([None for _ in range(max_len - len(c[k]))])

        df_time = pd.DataFrame(t, columns=[*[o.__name__ for o in optimizers], MultiStartLocalSearchOptimizer.__name__])
        df_cost = pd.DataFrame(c, columns=[*[o.__name__ for o in optimizers], MultiStartLocalSearchOptimizer.__name__])

        df_time = df_time.describe().loc[['min', 'mean', 'max']]
        df_cost = df_cost.describe().loc[['min', 'mean', 'max']]

        print('[TIME]')
        print(df_time)

        print('[COST]')
        print(df_cost)

        inform.write('[VISUALISING]\n')
        for method, bs in zip([MultiStartLocalSearchOptimizer.__name__,
                               *[o.__name__ for o in optimizers]], best_solutions_routes):
            visualize_route(bs.route, points, output_path, f'{fname}_{method}')

    inform.write('[FINISHED]\n')


if __name__ == '__main__':
    stdout = argv[4]
    inform = sys.stdout
    with open(stdout, 'w') as stdout:
        sys.stdout = stdout
        main(argv[1], int(argv[2]), argv[3])
