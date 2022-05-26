import os
from multiprocessing import Pool
from statistics import mean
from typing import Callable

from common.program_synthesis.dsl import DomainSpecificLanguage, StandardDomainSpecificLanguage
from common.environment.environment import Environment
from metasynthesis.performance_function.dom_dist_fun.robot_dist_fun import RobotDistFun
from metasynthesis.performance_function.evolving_function import distance_default_expr_tree
from metasynthesis.performance_function.expr_tree import ExpressionTree
from metasynthesis.performance_function.symbol import TermSym
from solver.runner.algorithms import dicts
from solver.runner.file_manager import FileManager
from solver.runner.test_case_retriever import get_test_cases


class Runner:
    # TODO implement DSL in Runner
    def __init__(self, lib,
                 algo: str,
                 setting: str,
                 test_cases: str,
                 time_limit_sec: float,
                 debug: bool = False,
                 store: bool = True,
                 suffix: str = "",
                 dsl: DomainSpecificLanguage = None,
                 dist_fun: Callable[[Environment, Environment], float] = None):
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

    def run(self):
        Runner.algo = self.algorithm

        if self.debug:
            print("Parsing test cases")

        exclude = self.file_manager.finished_test_cases() if self.store else []
        all_cases = get_test_cases(self.settings.domain, self.files, exclude)

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

            # with Pool(processes=1) as pool:
            with Pool(processes=os.cpu_count() - 1) as pool:
                results = pool.map_async(self.execute_test_case, cases)
                # if True:
                # results = [self.execute_test_case(c) for c in cases]

                stats_list = []
                for program, stats in results.get():
                    stats_list.append(stats)

                    total_examples[i] += stats["test_total"]
                    solved_examples[i] += stats["test_correct"]

                if self.store:
                    self.file_manager.append_result(stats_list)

        accuracies = [float(s) / t for s, t in zip(solved_examples, total_examples)]

        return mean(accuracies)

    def execute_test_case(self, test_case):
        return self.algorithm.run(self.settings, self.time_limit_sec, self.debug, test_case)


if __name__ == "__main__":
    time_limit = 1
    debug = False
    store = True  # sets whether to write date to file
    setting = "RO"
    algo = "Brute"
    test_cases = "debug"
    # params = [0, 0.1, 0.5, 1, 1.5, 2]

    # params = params if test_cases == "param" else [0]
    param = 0
    store = False if test_cases == "param" else store

    functions = RobotDistFun.partial_dist_funs()
    terms = list(map(lambda x: TermSym(x), functions))
    rand_obj_fun = ExpressionTree.generate_random_expression(terms=terms, max_depth=4)
    dist_fun = rand_obj_fun.distance_fun
    # dist_fun = distance_default_expr_tree
    print(dist_fun)
    mean1 = Runner(dicts(), algo, setting, test_cases, time_limit, debug, store, dist_fun=dist_fun).run()

    print(f"Solved {str(mean1)}")
