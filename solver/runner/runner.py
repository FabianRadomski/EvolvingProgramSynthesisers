import os
import time
from multiprocessing import Pool
from statistics import mean
from typing import Callable, Dict

from common.program_synthesis.dsl import DomainSpecificLanguage
from common.environment.environment import Environment
from metasynthesis.performance_function.dom_dist_fun.robot_dist_fun import RobotDistFun
from metasynthesis.performance_function.expr_tree import ExpressionTree
from metasynthesis.performance_function.symbol import TermSym

from common.program import Program
from solver.runner.algorithms import dicts
from solver.runner.file_manager import FileManager
from solver.runner.test_case_retriever import get_test_cases


class Runner:
    def __init__(self, lib,
                 algo: str,
                 setting: str,
                 test_cases: str,
                 time_limit_sec: float,
                 debug: bool = False,
                 store: bool = True,
                 suffix: str = "",
                 dsl: DomainSpecificLanguage = None,
                 dist_fun: Callable[[Environment, Environment], float] = None,
                 multi_thread: bool = True,
                 left_bound_cases: float = 0,
                 right_bound_cases: float = 1):
        self.time_limit_sec = time_limit_sec
        self.debug = debug
        self.store = store

        self.algorithm = lib["algorithms"][algo][setting]
        self.settings = lib["settings"][setting]

        if dsl is not None:
            self.settings.dsl = dsl

        self.settings.dist_fun = dist_fun

        self.files = lib["test_cases"][test_cases][setting[0]]

        self.file_manager = FileManager(algo, setting, suffix)
        self.multi_thread = multi_thread
        self.left_bound_cases = left_bound_cases
        self.right_bound_cases = right_bound_cases

        self.search_results = dict()

    def run(self):
        Runner.algo = self.algorithm

        if self.debug:
            print("Parsing test cases")

        exclude = self.file_manager.finished_test_cases() if self.store else []
        all_cases = get_test_cases(self.settings.domain, self.files, exclude)
        all_cases = all_cases[round(self.left_bound_cases * len(all_cases)):round(self.right_bound_cases*len(all_cases))]

        if self.debug:
            print("Parsed {} test cases".format(len(all_cases)))

        if len(all_cases) == 0:
            print("All test cases were already run")
            return 0

        total_examples = [0] * len(self.files[2])
        solved_examples = [0] * len(self.files[2])
        for i, trial in enumerate(self.files[2]):
            if self.debug:
                print("Running trial #{}...".format(trial))

            cases = [tc for tc in all_cases if tc.index[2] == trial]

            if len(cases) == 0:
                print("All test cases were already run")
                total_examples[i] = 0
                solved_examples[i] = -1
                continue

            stats_list = []

            if not self.multi_thread:
                for case in cases:
                    program, stats = self.execute_test_case(case)
                    total_examples[i] += stats["test_total"]
                    solved_examples[i] += stats["test_correct"]
                    stats_list.append(stats)

                    self.update_search_results(stats, program)
            else:
                # with Pool(processes=1) as pool:
                with Pool(processes=os.cpu_count() - 1) as pool:
                    results = pool.map_async(self.execute_test_case, cases)
                    # if True:
                    # results = [self.execute_test_case(c) for c in cases]

                    for program, stats in results.get():
                        stats_list.append(stats)

                        total_examples[i] += stats["test_total"]
                        solved_examples[i] += stats["test_correct"]

                        self.update_search_results(stats, program)

                if self.store:
                    self.file_manager.append_result(stats_list)

        accuracies = [float(s) / t for s, t in zip(solved_examples, total_examples)]

        return mean(accuracies)

    def update_search_results(self, stats: Dict, best_program: Program):
        self.search_results[str(stats["complexity"]) + str(stats["task"]) +
                            str(stats["trial"]) + str(self.settings.dsl)] = \
            {"best_program": best_program,
             "train_correct": stats["train_correct"],
             "test_correct": stats["test_correct"],
             "test_total": stats["test_total"],
             "search_time": stats["execution_time"]}

    def execute_test_case(self, test_case):
        return self.algorithm.run(self.settings, self.time_limit_sec, self.debug, test_case, best_program=Program([]))


if __name__ == "__main__":
    time_limit = 1
    debug = False
    store = False
    setting = "RG"
    algo = "Brute"
    test_cases = "param"
    params = [0, 0.1, 0.5, 1, 1.5, 2]

    params = params if test_cases == "param" else [0]
    store = False if test_cases == "param" else store

    functions = RobotDistFun.partial_dist_funs()
    terms = list(map(lambda x: TermSym(x), functions))
    rand_obj_fun = ExpressionTree.generate_random_expression(terms=terms, max_depth=4)
    dist_fun = rand_obj_fun.distance_fun

    for param in params:
        if test_cases == "param":
            print("\nParam = {}".format(param))

        mean1 = Runner(dicts(param), algo, setting, test_cases, time_limit, debug, store, dist_fun=dist_fun).run()
        print(f"Solved {str(mean1)}")
