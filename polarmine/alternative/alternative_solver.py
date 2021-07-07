from typing import List
from abc import ABC, abstractmethod
from polarmine.graph import InteractionGraph


class AlternativeSolver(ABC):
    """A solver for an alternative formulation of Echo Chamber Problem"""

    def __init__(self, *args, **kwargs):
        super(AlternativeSolver, self).__init__(*args, **kwargs)

    @abstractmethod
    def solve(
        self, graph: InteractionGraph, alpha: float
    ) -> tuple[float, List[int]]:
        pass
