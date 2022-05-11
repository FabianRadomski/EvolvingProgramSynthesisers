import os
import time
from dataclasses import dataclass
# takes as input a DSL,
from multiprocessing import Pool
from typing import List

from common.experiment import TestCase
from common.program import Program
from common.program_synthesis.dsl import DomainSpecificLanguage, StandardDomainSpecificLanguage
from common.program_synthesis.objective import ObjectiveFun
from example_parser.parser import Parser
from example_parser.pixel_parser import PixelParser
from example_parser.robot_parser import RobotParser
from example_parser.string_parser import StringParser
from search.abstract_search import SearchAlgorithm
from search.search_result import SearchResult
from search.brute.brute import Brute


@dataclass
class Runner:
    """Runner for running a program synthesizer for a given domain specific language NOT FOR a meta-synthesizer"""
    domain = "robot"
    dsl: DomainSpecificLanguage = StandardDomainSpecificLanguage(domain)
    search_method: SearchAlgorithm = Brute(5, ObjectiveFun(domain).fun)
    print_results: bool = False
    max_test_cases: int = 1000
    POOL_RUN_PROCESS_TIMEOUT = 10  # Must be higher than MAX_EXECUTION_TIME
    MULTI_PROCESS = True
    NO_PROCESSES = os.cpu_count() - 1

    # Create experiment runner using specified search and DSL
    def run(self):

        sum_of_success_percentages = 0
        sum_of_execution_times_in_seconds = 0
        number_of_completely_successful_programs = 0

        # extract tokens from the experiment's domain name
        test_cases = self._get_test_cases()

        results = []
        if self.MULTI_PROCESS:
            with Pool(processes=self.NO_PROCESSES) as pool:
                for tc in test_cases:
                    result = pool.apply_async(self.run_single_test_case, args=(tc,))
                    results.append(result)
                new_results = []
                for r in results:
                    try:
                        result = r.get(timeout=self.POOL_RUN_PROCESS_TIMEOUT)
                        new_results.append(result)
                    except:  # TimeoutError
                        continue
                results = new_results
        else:
            for tc in test_cases:
                result = self.run_single_test_case(tc)
                results.append(result)

        search_results = []
        for result in results:
            success_percentage, execution_time_in_seconds, search_result = result
            sum_of_success_percentages += success_percentage
            sum_of_execution_times_in_seconds += execution_time_in_seconds
            if success_percentage == 100.0:
                number_of_completely_successful_programs += 1
            search_results.append(search_result)

        average_success_percentage = sum_of_success_percentages / len(test_cases)
        average_execution_time = sum_of_execution_times_in_seconds / len(test_cases)
        percentage_of_completely_successful_programs = number_of_completely_successful_programs / len(test_cases) * 100

        return {
            "average_success": average_success_percentage,
            "average_execution": average_execution_time,
            "completely_successful_percentage": percentage_of_completely_successful_programs,
            "programs": search_results
        }

    def _instantiate_parser(self) -> Parser:
        if self.dsl.domain_name == "pixel":
            return PixelParser()
        if self.dsl.domain_name == "robot":
            return RobotParser()
        if self.dsl.domain_name == "string":
            return StringParser()

    def _get_test_cases(self) -> List[TestCase]:
        """Gets all test cases for the runner domain"""
        parser = self._instantiate_parser()
        test_cases = []
        directory = parser.path

        num_test_cases = 0
        for file in os.listdir(directory):
            if num_test_cases > self.max_test_cases:
                break
            test_cases.append(parser.parse_file(file))
            num_test_cases += 1

        return test_cases

    def run_single_test_case(self, test_case: TestCase):
        start_time = time.time()

        # # find program that satisfies training_examples
        search_result: SearchResult = self.search_method.run(test_case.training_examples, self.dsl.get_trans_tokens(),
                                   self.dsl.get_bool_tokens())

        if self.print_results:
            self.print_info(self.get_result_info(test_case, search_result))

        program: Program = search_result.dictionary["program"]

        execution_time_in_seconds = time.time() - start_time

        successes = 0

        for e in test_case.test_examples:
            in_state = e.input_environment
            out_state = e.output_environment

            try:
                result = program.interp(in_state)
            except:
                # print("interpreting the program threw an error")
                result = in_state

            if out_state.correct(result):
                successes += 1

        success_percentage = 100.0 * successes / len(test_case.test_examples)
        return success_percentage, execution_time_in_seconds, search_result

    @staticmethod
    def get_result_info(test_case: TestCase, result: SearchResult) -> dict:
        result_dict = result.dictionary
        file_path = test_case.path_to_result_file.split("-")
        program = result_dict["program"]

        info = {
            "file": "{}-{}-{}".format(file_path[1], file_path[2], file_path[3]),
            "test_cost": SearchAlgorithm.cost(test_case.test_examples, program),
            "train_cost": SearchAlgorithm.cost(test_case.training_examples, program),
            "execution_time": result_dict["execution_time"],
            "program_length": result_dict["program_length"],
            "iterations": result_dict["number_of_iterations"]
        }

        return info

    @staticmethod
    def print_info(result_info: dict):
        print(result_info)

"""
Example for running a test with the runner:
"""
if __name__ == '__main__':
    domain = "robot"
    data = Runner(StandardDomainSpecificLanguage(domain), Brute(10, ObjectiveFun(domain).fun)).run()
    print(data)
