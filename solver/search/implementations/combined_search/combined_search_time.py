from typing import List, Tuple, Type

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

    def update_info(self):
        """
        Updates the current information about the state of search with the information from the current search procedure.
        Should only be called once per search procedure.
        """
        self.statistics["no._explored_programs"] += self.current_algorithm.statistics["no._explored_programs"]
        self.best_program = self.current_algorithm.best_program
        self.statistics["best_cost_per_iteration"].extend(
            [(it + self.statistics["no._iterations"], cost) for it, cost in self.current_algorithm.statistics["best_cost_per_iteration"]])
