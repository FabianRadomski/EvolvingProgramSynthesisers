from common.program import Program
from search.abstract_search import SearchAlgorithm


class CombinedSearch(SearchAlgorithm):
    def __init__(self, time_limit_sec: float,
                 iterations_limit: int = 0,
                 best_program: Program = Program([])):
        super().__init__(time_limit_sec, iterations_limit=iterations_limit, best_program=best_program)
