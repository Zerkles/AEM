from collections import defaultdict
from typing import List

import pandas as pd
from sys import argv

from numpy import cov
from scipy.stats import pearsonr

from loader import iterate_instances
from time import time
import random
import numpy as np
import matplotlib.pyplot as plt
import sys

from optimizers import Route, MultiStartLocalSearchOptimizer
from optimizers.base import Solution
from optimizers.ils_optimizer.ils_optimizer import ILS1, ILS2b, ILS2a
from p6.optimizers.global_convexity.global_convexity_optimizer import GlobalConvexityOptimizer

pd.options.display.max_columns = None
pd.options.display.max_rows = None


def visualize(data, filename, path):
    plt.title(filename)

    x_cost = [x[0] for x in data]
    y_best = [x[1] for x in data]
    y_avg = [x[2] for x in data]
    print(filename + "| Coefficients Cost-Best:", pearsonr(x_cost, y_best))
    print(filename + "| Coefficients Cost-Average:", pearsonr(x_cost, y_avg))

    plt.plot(x_cost, y_best, label="Similarity to best solution")
    plt.plot(x_cost, y_avg, label="Average similarity to other solutions")
    plt.xlabel("Target function value")
    plt.xlabel("% of similarity")
    plt.legend()
    plt.savefig(path + '/' + filename)
    plt.clf()


def main(instances_path: str, repeat: int, output_path: str):
    print('[STARTED]')

    for x in iterate_instances(instances_path):
        fname, distance_matrix, points = x.name, x.distance_matrix, x.points

        print('[INSTANCE]', fname)
        inform.write(f'[INSTANCE] {fname}\n')

        # vertices = distance_matrix.shape[0]

        v_sim, e_sim = GlobalConvexityOptimizer(distance_matrix, points)()

        visualize(v_sim, fname + "_VertexGlobalConvexity", output_path)
        visualize(e_sim, fname + "_EdgeGlobalConvexity", output_path)

    inform.write('[FINISHED]\n')


if __name__ == '__main__':
    stdout = argv[4]
    inform = sys.stdout
    with open(stdout, 'w') as stdout:
        sys.stdout = stdout
        main(argv[1], int(argv[2]), argv[3])
