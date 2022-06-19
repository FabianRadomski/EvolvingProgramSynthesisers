import os
import time
from multiprocessing import Pool
from statistics import mean
from typing import Callable
import numpy as np

from common.environment.environment import Environment
from metasynthesis.performance_function.dom_dist_fun.robot_dist_fun import RobotDistFun
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
                 dist_fun: Callable[[Environment, Environment], float] = None):
        self.time_limit_sec = time_limit_sec
        self.debug = debug
        self.store = store

        self.algorithm = lib["algorithms"][algo][setting]
        self.settings = lib["settings"][setting]
        self.settings.dist_fun = dist_fun
        self.files = lib["test_cases"][test_cases][setting[0]]

        self.file_manager = FileManager(algo, setting, suffix)

    def run(self, alternative_fitness=False):
        if alternative_fitness:
            Runner.algo = self.algorithm

            if self.debug:
                print("Parsing test cases")

            exclude = self.file_manager.finished_test_cases() if self.store else []
            all_cases = get_test_cases(self.settings.domain, self.files, exclude)

            if self.debug:
                print("Parsed {} test cases".format(len(all_cases)))

            if len(all_cases) == 0:
                print("All test cases were already run")
                return 0, 0

            total_examples = [0] * len(self.files[1])
            solved_examples = [0] * len(self.files[1])

            # print('herehrher:', self.files[2])
            for i, task in enumerate(self.files[1]):
                # print(i, ',,,', trial)
                if self.debug:
                    print("Running task #{}...".format(task))

                cases = [tc for tc in all_cases if tc.index[1] == task]

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
                    normalized_run_times = []
                    for program, stats in results.get():
                        stats_list.append(stats)
                        normalized_run_times.append(
                            min(self.time_limit_sec, stats["execution_time"]) / self.time_limit_sec)

                        total_examples[i] += stats["test_total"]
                        solved_examples[i] += stats["test_correct"]

                    if self.store:
                        self.file_manager.append_result(stats_list)

            num_solved_tasks = 0
            num_ex_for_unsolved_tasks = []
            num_unsolved_ex_for_unsolved_tasks = []
            for i in range(len(self.files[1])):
                if solved_examples[i] == total_examples[i]:
                    num_solved_tasks += 1
                else:
                    num_ex_for_unsolved_tasks.append(total_examples[i])
                    num_unsolved_ex_for_unsolved_tasks.append(total_examples[i] - solved_examples[i])
            percentage_of_solved_tasks = num_solved_tasks / len(self.files[1])
            # print('percentage_of_solved_tasks:', percentage_of_solved_tasks)
            mean_perc_unsolved_ex_unsolved_tasks = mean([float(s) / t for s, t in
                                                    zip(num_unsolved_ex_for_unsolved_tasks, num_ex_for_unsolved_tasks)])
            # print('mean percentage of unsolved examples for unsolved tasks:', mean_perc_unsolved_ex_unsolved_tasks)
            # accuracies = [float(s) / t for s, t in zip(solved_examples, total_examples)]
            average_norm_run_time = mean(normalized_run_times)
            return percentage_of_solved_tasks, mean_perc_unsolved_ex_unsolved_tasks, average_norm_run_time
        else:
            Runner.algo = self.algorithm

            if self.debug:
                print("Parsing test cases")

            exclude = self.file_manager.finished_test_cases() if self.store else []
            all_cases = get_test_cases(self.settings.domain, self.files, exclude)

            if self.debug:
                print("Parsed {} test cases".format(len(all_cases)))

            if len(all_cases) == 0:
                print("All test cases were already run")
                return 0, 0

            total_examples = [0] * len(self.files[2])
            solved_examples = [0] * len(self.files[2])

            # print('herehrher:', self.files[2])
            for i, trial in enumerate(self.files[2]):
                # print(i, ',,,', trial)
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
                    normalized_run_times = []
                    for program, stats in results.get():
                        stats_list.append(stats)
                        normalized_run_times.append(
                            min(self.time_limit_sec, stats["execution_time"]) / self.time_limit_sec)

                        total_examples[i] += stats["test_total"]
                        solved_examples[i] += stats["test_correct"]

                    if self.store:
                        self.file_manager.append_result(stats_list)

            accuracies = [float(s) / t for s, t in zip(solved_examples, total_examples)]
            average_norm_run_time = mean(normalized_run_times)
            return mean(accuracies), average_norm_run_time



    def execute_test_case(self, test_case):
        res = self.algorithm.run(self.settings, self.time_limit_sec, self.debug, test_case)
        return res


if __name__ == "__main__":
    time_limit = 1
    debug = False
    store = True  # sets wether to write date to file
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
