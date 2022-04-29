import time
from statistics import mean
from typing import List

from common.environment import StringEnvironment
from common.experiment import Example
from common.prorgam import Program
from common.tokens.abstract_tokens import EnvToken, InvalidTransition
from common.tokens.control_tokens import LoopIterationLimitReached
from search.search_result import SearchResult


class SearchAlgorithm:
    """Abstract interface for a program synthesis search algorithm."""

    def __init__(self, time_limit_sec: float, iterations_limit: int = 0, best_program: Program = Program([])):
        self.time_limit_sec = time_limit_sec
        self.iterations_limit = iterations_limit
        self._best_program = best_program
        self.number_of_explored_programs = 0
        self.cost_per_iteration = [(0, float("inf"))]  # save (iteration_number, cost) when new best_program is found
        self.number_of_iterations = 0

    @property
    def best_program(self) -> Program:
        return self._best_program

    def setup(self, training_examples: List[Example], trans_tokens: list[EnvToken], bool_tokens: list[EnvToken]):
        """This method is called before a search is performed. The search will be performed for the given
        'training_examples'. Also the 'trans_tokens' and 'bool_tokens' that are available for the environment are given.
        """

        raise NotImplementedError()

    def iteration(self, training_example: List[Example], trans_tokens: list[EnvToken], bool_tokens: list[EnvToken]) -> bool:
        """This method represents an iteration of the search algorithm. This method will get called over and over 
        again, as long as it returns True. It will stop whenever False is returned or a time limit is reached. The 
        search will be performed for the given 'training_examples'. Also the 'trans_tokens' and 'bool_tokens' that are
        available for the environment are given."""

        raise NotImplementedError()

    def extend_result(self, search_result: SearchResult):
        """The result of a search is stored a SearchResult object, containing a key-value dictionary. Whenever an
        algorithm needs to collect more data than is collected by default, one can append key-value pairs to the result.
        """

        return search_result

    def run(self, training_examples: List[Example], trans_tokens: list[EnvToken], bool_tokens: list[EnvToken]) -> SearchResult:
        """"Runs the search method until a program is returned or the time limit is reached. First the setup method is
        called, followed by a repetition of the iteration method until either a result is obtained, or the time limit is
        reached"""
        start_time = time.process_time()

        # Reset String distance dictionary
        StringEnvironment.distance_map = {}

        # Call setup.
        self.setup(training_examples, trans_tokens, bool_tokens)

        # If no iteration limit was set, use the time limit.
        if self.iterations_limit == 0:
            # self.iteration returns whether a new iteration should be performed. Break the loop if time limit reached.
            while self.iteration(training_examples, trans_tokens, bool_tokens):
                if time.process_time() >= start_time + self.time_limit_sec:
                    break
        else:
            # Iterate until the iteration limit is reached.
            n_iteration = 0
            while self.iteration(training_examples, trans_tokens, bool_tokens):
                if n_iteration >= self.iterations_limit:
                    break
                n_iteration += 1

        run_time = time.process_time() - start_time

        # Extend results and return.
        return self.extend_result(SearchResult(
            program=self.best_program,
            process_time_sec=run_time,
            number_of_explored_programs=self.number_of_explored_programs,
            cost_per_iteration=self.cost_per_iteration,
            number_of_iterations=self.number_of_iterations
        ))

    @staticmethod
    def cost(exs: list[Example], p: Program):
        def ex_cost(ex: Example):
            try:
                return p.interp(ex.input_environment).distance(ex.output_environment)
            except (InvalidTransition, LoopIterationLimitReached):
                return float('inf')

        return mean([ex_cost(ex) for ex in exs])
