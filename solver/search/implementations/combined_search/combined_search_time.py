import time
from typing import List, Tuple, Type

from common.experiment import TestCase
from common.program import Program
from common.settings.settings import Settings
from metasynthesis.search_procedure.alg_init_dict import alg_init_dict
from solver.search.search_algorithm import SearchAlgorithm


# AStar: "AS", Brute: "Brute", MCTS: "MCTS", LNS: "LNS", MetropolisHasting: "MH", GeneticAlgorithm: "GP"
class CombinedSearchTime(SearchAlgorithm):
    def __init__(self, algorithm_sequence: List[Tuple[str, int]], setting: str):
        # Each search class is associated with the name from the dicts class in algorithms.py

        # dictionary containing the settings
        self.running_info: dict = alg_init_dict()
        self.algorithm_sequence: List[Tuple[str, int]] = algorithm_sequence
        self.sequence_pos: int = 0  # which position of the algorithm sequence is currently executed

        self.setting = setting
        self.curr_time: float = 0

        self.current_algorithm: SearchAlgorithm = self.running_info[self.algorithm_sequence[0][0]][self.setting]

    def setup(self):
        self.sequence_pos = 0
        self.curr_time = 0
        self.best_program = Program([])

    def iteration(self) -> bool:
        """
         If the search is based on time instead of iterations, this function runs next search procedure in chain.
        """

        self.current_algorithm: SearchAlgorithm = self.running_info[self.algorithm_sequence[self.sequence_pos][0]][self.setting]

        self.current_algorithm.run(settings=self.settings, debug=False, time_limit_sec=self.algorithm_sequence[self.sequence_pos][1],
                                   test_case=self.test_case, best_program=self.best_program)

        self.sequence_pos += 1

        self.update_info()

        # Check if the cost of the last iteration is zero (the solution is found)
        if self.current_algorithm.statistics["best_cost_per_iteration"][-1][1] == 0 or self.sequence_pos == len(self.algorithm_sequence):
            return False
        return True

    def run(self, settings: Settings, time_limit_sec: float, debug: bool, test_case: TestCase, best_program: Program = Program([])) -> (Program, dict):
        """"Runs the solver method until a program is returned or the time limit is reached. First the setup method is
        called, followed by a repetition of the iteration method until either a result is obtained, or the time limit is
        reached"""
        start_time = time.process_time()

        self.settings = settings
        self.time_limit_sec = time_limit_sec

        self.test_case = test_case
        self.training_examples = test_case.training_examples
        self.input_state = tuple([t.input_environment for t in self.training_examples])
        self.test_examples = test_case.test_examples

        self.debug = debug


        self.best_program = best_program

        self.statistics = {
            "complexity": test_case.index[0],
            "task": test_case.index[1],
            "trial": test_case.index[2],
            "no._explored_programs": 1,
            "best_cost_per_iteration": [],
            "no._iterations": 0,
        }

        self.setup()

        # self.iteration returns whether a new iteration should be performed. Break the loop if time limit reached.
        while self.iteration():
            self.statistics["no._iterations"] += 1

            if time.process_time() >= start_time + self.time_limit_sec:
                break

        run_time = time.process_time() - start_time

        self.statistics["execution_time"] = run_time
        self.statistics["best_program"] = str(self.best_program)
        self.statistics["test_cost"], _, self.statistics["test_correct"] = self.current_algorithm.evaluate(self.best_program, train=False)
        self.statistics["train_cost"], _, self.statistics["train_correct"] = self.current_algorithm.evaluate(self.best_program, train=True)
        self.statistics["test_total"] = len(self.test_examples)

        if self.debug:
            print(self.__class__.__name__ + "\n")
            print(self.statistics)

        # Extend results and return.
        return self.best_program, self.statistics

    def update_info(self):
        """
        Updates the current information about the state of search with the information from the current search procedure.
        Should only be called once per search procedure.
        """
        self.statistics["no._explored_programs"] += self.current_algorithm.statistics["no._explored_programs"]
        self.best_program = self.current_algorithm.best_program
        self.statistics["best_cost_per_iteration"].extend(
            [(it + self.statistics["no._iterations"], cost) for it, cost in self.current_algorithm.statistics["best_cost_per_iteration"]])
