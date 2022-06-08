import os
from multiprocessing import Pool
from statistics import mean
from typing import List, Tuple

import numpy as np

from common.program import Program
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
                 multi_thread: bool = True):
        self.time_limit_sec = time_limit_sec
        self.debug = debug
        self.store = store

        self.algorithm = lib["algorithms"][algo][setting]
        self.settings = lib["settings"][setting]
        self.files = lib["test_cases"][test_cases][setting[0]]

        self.file_manager = FileManager(algo, setting, suffix)
        self.multi_thread = multi_thread

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
                with Pool(processes=os.cpu_count() - 1) as pool:
                    results = pool.map_async(self.execute_test_case, cases)
                    # if True:
                    # results = [self.execute_test_case(c) for c in cases]

                    stats_list = []
                    for program, stats in results.get():
                        stats_list.append(stats)

                        total_examples[i] += stats["test_total"]
                        solved_examples[i] += stats["test_correct"]
                        run_time += stats["execution_time"]

                    if self.store:
                        self.file_manager.append_result(stats_list)

        accuracies = [float(s) / t for s, t in zip(solved_examples, total_examples)]
        average_time = run_time / np.sum(total_examples)

        return mean(accuracies), average_time, stats_list

    def execute_test_case(self, test_case):
        return self.algorithm.run(self.settings, self.time_limit_sec, self.debug, test_case, best_program = Program([]))


if __name__ == "__main__":
    time_limit = 0.2
    debug = False
    store = False
    setting = "RO"
    algo = "MH"
    test_cases = "eval"

    store = False if test_cases == "param" else store

    mean1 = Runner(dicts(0), algo, setting, test_cases, time_limit, debug, store).run()

    print(f"Solved {str(mean1)}")
