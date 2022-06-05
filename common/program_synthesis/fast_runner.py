import os
from multiprocessing import Pool
from statistics import mean
from typing import List, Tuple

import numpy as np

from common.program import Program
from common.settings.settings import Settings
from solver.runner.algorithms import dicts
from solver.runner.file_manager import FileManager
from solver.runner.test_case_retriever import get_test_cases
from solver.search.search_algorithm import SearchAlgorithm


class Runner:
    # TODO implement DSL in Runner
    def __init__(self, alg_name: str,
                 setting_name: str,
                 algorithm: SearchAlgorithm,
                 settings: Settings,
                 test_cases: Tuple,
                 time_limit_sec: float,
                 debug: bool = False,
                 store: bool = True,
                 suffix: str = "",
                 multi_thread: bool = True):
        self.time_limit_sec = time_limit_sec
        self.debug = debug
        self.store = store

        self.algorithm = algorithm
        self.settings = settings
        self.files = test_cases

        self.file_manager = FileManager(alg_name, setting_name, suffix)
        self.multi_thread = multi_thread

        self.NO_PROCESSES = 4
        self.POOL_RUN_PROCESS_TIMEOUT = 10

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
            return 0, 0

        total_examples = [0] * len(self.files[2])
        solved_examples = [0] * len(self.files[2])
        run_time = 0
        for i, trial in enumerate(self.files[2]):
            if self.debug:
                print("Running trial #{}...".format(trial))

            cases = [tc for tc in all_cases if tc.index[2] == trial]

            if len(cases) == 0:
                print("All test cases were already run")
                total_examples[i] = 0
                solved_examples[i] = -1
                continue
            if not self.multi_thread:
                stats_list = []
                for case in cases:
                    program, stats = self.execute_test_case(case)
                    total_examples[i] += stats["test_total"]
                    solved_examples[i] += stats["test_correct"]
                    run_time += stats["execution_time"]

                    stats_list.append(stats)
                    if self.store:
                        self.file_manager.append_result(stats_list)

            else:
                # with Pool(processes=1) as pool:
                results = []
                with Pool(processes=self.NO_PROCESSES) as pool:
                    for tc in cases:
                        result = pool.apply_async(self.execute_test_case(tc), args=(tc,))
                        results.append(result)
                    stats_list = []
                    for r in results:
                        try:
                            program, stats = r.get(timeout=self.POOL_RUN_PROCESS_TIMEOUT)

                            stats_list.append(stats)
                            total_examples[i] += stats["test_total"]
                            solved_examples[i] += stats["test_correct"]
                            run_time += stats["execution_time"]
                        except:  # TimeoutError
                            continue
                    if self.store:
                        self.file_manager.append_result(stats_list)

                # with Pool(processes=os.cpu_count() - 1) as pool:
                #     results = pool.map_async(self.execute_test_case, cases)
                #     # if True:
                #     # results = [self.execute_test_case(c) for c in cases]
                #
                #     stats_list = []
                #     for program, stats in results.get():
                #         stats_list.append(stats)
                #
                #         total_examples[i] += stats["test_total"]
                #         solved_examples[i] += stats["test_correct"]
                #         run_time += stats["execution_time"]
                #
                #     if self.store:
                #         self.file_manager.append_result(stats_list)

        accuracies = [float(s) / t for s, t in zip(solved_examples, total_examples)]
        average_time = run_time / np.sum(total_examples)

        return mean(accuracies), average_time

    def execute_test_case(self, test_case):
        return self.algorithm.run(self.settings, self.time_limit_sec, self.debug, test_case, best_program=Program([]))


if __name__ == "__main__":
    time_limit = 10
    debug = False
    store = True  # sets wether to write date to file
    setting = "RE"
    algo = "AS"
    test_cases = "debug"
    # params = [0, 0.1, 0.5, 1, 1.5, 2]

    # params = params if test_cases == "param" else [0]
    param = 0
    store = False if test_cases == "param" else store

    mean1, run_time = Runner(dicts(), algo, setting, test_cases, time_limit, debug, store).run()

    print(f"Solved {str(mean1)}")
