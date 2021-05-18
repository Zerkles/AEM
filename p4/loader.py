from csv import reader
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, NewType
import numpy as np

Matrix2D = NewType('Matrix2D', np.ndarray)


@dataclass
class Instance:
    name: str
    points: List[List[int]]
    distance_matrix: Matrix2D


def iterate_instances(instances_path: str) -> Generator[Instance, None, None]:
    instances_path = Path(instances_path)

    for instance in instances_path.iterdir():
        with instance.open('r') as f_instance:
            r = reader(f_instance, delimiter=' ')
            instance_name: str = next(r)[1].split('.')[0]
            points = [[int(strnum) for strnum in line] for line in r if line[0].isnumeric()]
            distance_matrix = __create_distance_matrix(points)

            yield Instance(instance_name, points, distance_matrix)


def __create_distance_matrix(points: List[List[int]]) -> Matrix2D:
    neighbors_amount = points[-1][0]
    neighbors_matrix = np.ndarray(shape=(neighbors_amount, neighbors_amount))

    # create & init distance/neighbors matrix
    for v_id in range(neighbors_amount):
        vertex = {'x': points[v_id][1], 'y': points[v_id][2]}
        for neighbor_id in range(neighbors_amount):
            if v_id == neighbor_id:
                neighbors_matrix[v_id][neighbor_id] = np.inf
            else:
                neighbor_vertex = {'x': points[neighbor_id][1], 'y': points[neighbor_id][2]}
                x_diff = vertex['x'] - neighbor_vertex['x']
                y_diff = vertex['y'] - neighbor_vertex['y']
                distance = np.round(np.sqrt(x_diff ** 2 + y_diff ** 2))
                neighbors_matrix[v_id][neighbor_id] = neighbors_matrix[neighbor_id][v_id] = distance

    return Matrix2D(neighbors_matrix)
