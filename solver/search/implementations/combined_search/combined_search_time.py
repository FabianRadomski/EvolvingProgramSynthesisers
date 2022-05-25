import time
from typing import List, Tuple, Type

from common.environment import StringEnvironment
from common.experiment import Example
from common.program import Program
from common.tokens.abstract_tokens import EnvToken
from search.abstract_search import SearchAlgorithm
from search.search_result import SearchResult


class CombinedSearchTime(SearchAlgorithm):
    def __init__(self, time_limit_sec: float,
                 algorithm_sequence: List[Tuple[Type[SearchAlgorithm], int]],
                 best_program: Program = Program([])):
        super().__init__(time_limit_sec, best_program=best_program)

        self.algorithm_sequence: List[Tuple[Type[SearchAlgorithm], int]] = algorithm_sequence
        self.sequence_pos: int = 0  # which position of the algorithm sequence is currently executed
        self.current_algorithm: SearchAlgorithm = self.algorithm_sequence[self.sequence_pos][0](self.algorithm_sequence[self.sequence_pos][1])

        self.training_examples: List[Example] = []
        self.trans_tokens: List[EnvToken] = []
        self.bool_tokens: list[EnvToken] = []

        self.curr_time: float = 0

    def setup(self, training_examples: List[Example], trans_tokens: list[EnvToken], bool_tokens: list[EnvToken]):
        self.training_examples = training_examples
        self.trans_tokens = trans_tokens
        self.bool_tokens = bool_tokens

        self.number_of_explored_programs = 0
        self.number_of_iterations = 0
        self._best_program = Program([])

    def iteration(self, training_example: List[Example], trans_tokens: list[EnvToken], bool_tokens: list[EnvToken]) -> bool:
        """
         If the search is based on time instead of iterations, this function runs next search procedure in chain.
        """

        self.current_algorithm = self.algorithm_sequence[self.sequence_pos][0](self.algorithm_sequence[self.sequence_pos][1],
                                                                               best_program=self.best_program)

        self.current_algorithm.run(self.training_examples, self.trans_tokens, self.bool_tokens)

        self.sequence_pos += 1
        self.number_of_iterations += 1

        self.update_info()

        if self.current_algorithm.cost_per_iteration[-1][1] == 0:
            return False
        return True

    def run(self, training_examples: List[Example], trans_tokens: list[EnvToken], bool_tokens: list[EnvToken]) -> SearchResult:
        start_time = time.process_time()

        # Reset String distance dictionary
        StringEnvironment.distance_map = {}

        # Call setup

        self.setup(training_examples, trans_tokens, bool_tokens)

        while self.iteration(training_examples, trans_tokens, bool_tokens):
            continue

        run_time = time.process_time() - start_time

        # Extend results and return.
        return self.extend_result(SearchResult(
            program=self.best_program,
            process_time_sec=run_time,
            number_of_explored_programs=self.number_of_explored_programs,
            cost_per_iteration=self.cost_per_iteration,
            number_of_iterations=self.number_of_iterations
        ))

    def update_info(self):
        """
        Updates the current information about the state of search with the information from the current search procedure.
        Should only be called once per search procedure.
        """
        self.number_of_explored_programs += self.current_algorithm.number_of_explored_programs
        self._best_program = self.current_algorithm.best_program
        self.cost_per_iteration.append([(it + self.number_of_iterations, cost) for it, cost in self.current_algorithm.cost_per_iteration])
