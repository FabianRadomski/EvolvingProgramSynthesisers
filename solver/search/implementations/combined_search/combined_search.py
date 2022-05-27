import time
from typing import List, Tuple, Type

from common.program import Program
from common.tokens.abstract_tokens import EnvToken
from solver.search.search_algorithm import SearchAlgorithm


class CombinedSearch(SearchAlgorithm):
    def __init__(self, time_limit_sec: float,
                 algorithm_sequence: List[Tuple[Type[SearchAlgorithm], int]],
                 iterations_limit: int = 0,
                 best_program: Program = Program([])):
        super().__init__(time_limit_sec, iterations_limit=iterations_limit, best_program=best_program)

        # assert that the iteration_limit is equal to the sum of all iterations of the algorithms
        if iterations_limit == 0:
            self.iterations_limit = sum(iter for alg, iter in algorithm_sequence)

        assert self.iterations_limit == sum(iter for alg, iter in algorithm_sequence)

        self.algorithm_sequence: List[Tuple[Type[SearchAlgorithm], int]] = algorithm_sequence
        self.current_algorithm: SearchAlgorithm = self.algorithm_sequence[0][0](time_limit_sec, iterations_limit=algorithm_sequence[0][1])
        self.sequence_pos: int = 0  # which position of the algorithm sequence is currently executed
        self.next_switch: int = 0  # at which iteration the next algorithm should be used

        self.training_examples: List[Example] = []
        self.trans_tokens: List[EnvToken] = []
        self.bool_tokens: list[EnvToken] = []

    def setup(self, training_examples: List[Example], trans_tokens: list[EnvToken], bool_tokens: list[EnvToken]):
        self.training_examples = training_examples
        self.trans_tokens = trans_tokens
        self.bool_tokens = bool_tokens

        self.number_of_explored_programs = 0
        self.number_of_iterations = 0
        self._best_program = Program([])

        self.current_algorithm: SearchAlgorithm = self.algorithm_sequence[0][0](self.time_limit_sec, iterations_limit=self.algorithm_sequence[0][1])
        self.current_algorithm.setup(training_examples, trans_tokens, bool_tokens)
        self.next_switch = self.algorithm_sequence[0][1]

    def iteration(self, training_example: List[Example], trans_tokens: list[EnvToken], bool_tokens: list[EnvToken]) -> bool:
        if self.number_of_iterations == self.next_switch:
            if not self.switch_algorithm():
                return False
            self.current_algorithm.setup(training_example, trans_tokens, bool_tokens)

        self.number_of_iterations += 1
        return self.current_algorithm.iteration(training_example, trans_tokens, bool_tokens)

    def run(self, training_examples: List[Example], trans_tokens: list[EnvToken], bool_tokens: list[EnvToken]) -> SearchResult:
        start_time = time.process_time()

        # Reset String distance dictionary
        StringEnvironment.distance_map = {}

        # Call setup
        self.setup(training_examples, trans_tokens, bool_tokens)

        while self.iteration(training_examples, trans_tokens, bool_tokens):
            continue
        self.update_info()

        run_time = time.process_time() - start_time

        # Extend results and return.
        return self.extend_result(SearchResult(
            program=self.best_program,
            process_time_sec=run_time,
            number_of_explored_programs=self.number_of_explored_programs,
            cost_per_iteration=self.cost_per_iteration,
            number_of_iterations=self.number_of_iterations
        ))

    def switch_algorithm(self) -> bool:
        """
        Switches algorithm to the next one in the chain. Returns false if the chain is all executed, true otherwise.
        """
        # TODO: Don't switch if the next algorithm is the same as current, just add iterations
        # Propagate in the chain of algorithms
        self.sequence_pos += 1
        if self.sequence_pos >= len(self.algorithm_sequence):
            return False

        # Update the current best program with the best program from the algorithm
        self.update_info()

        # Update the current algorithm and the next_switch
        self.current_algorithm = self.algorithm_sequence[self.sequence_pos][0](self.time_limit_sec, best_program=self.best_program)
        self.current_algorithm.setup(self.training_examples, self.trans_tokens, self.bool_tokens)
        self.next_switch = self.algorithm_sequence[self.sequence_pos][1] + self.number_of_iterations
        return True

    def update_info(self):
        """
        Updates the current information about the state of search with the information from the current search procedure.
        Should only be called once per search procedure.
        """
        self.number_of_explored_programs += self.current_algorithm.number_of_explored_programs
        self._best_program = self.current_algorithm.best_program
        self.cost_per_iteration.append([(it + self.number_of_iterations, cost) for it, cost in self.current_algorithm.cost_per_iteration])
