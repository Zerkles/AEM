from abc import abstractmethod
from time import time
from typing import Generator

from loader import Matrix2D
from .base import Optimizer, Route, Solution


class TimerOptimizer(Optimizer):
    def __init__(self, distance_matrix: Matrix2D, route: Route, max_processing_time_s: float):
        super().__init__(distance_matrix, route)
        self.max_processing_time_s = max_processing_time_s

    def _search(self) -> Solution:
        data = None
        start = time()
        find_solution = self._find_solution()
        while time() - start < self.max_processing_time_s:
            data = next(find_solution)
        return data

    @abstractmethod
    def _find_solution(self) -> Generator[Solution, None, None]:
        pass
