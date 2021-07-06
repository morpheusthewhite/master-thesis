from typing import List
from abc import ABC, abstractmethod
from polarmine.graph import InteractionGraph


class ECPSolver(ABC):
    """A solver for the Echo Chamber Problem"""

    def __init__(self, *args, **kwargs):
        super(ECPSolver, self).__init__(*args, **kwargs)

    @abstractmethod
    def solve(
        self, graph: InteractionGraph, alpha: float
    ) -> tuple[float, List[int], List[int], List[str]]:
        pass
