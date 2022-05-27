import os
import time
from dataclasses import dataclass, field
# takes as input a DSL,
from multiprocessing import Pool
from typing import Type, List

from common.experiment import TestCase, Experiment
from common.program import Program
from common.program_synthesis.dsl import DomainSpecificLanguage, StandardDomainSpecificLanguage
from example_parser.parser import Parser
from example_parser.pixel_parser import PixelParser
from example_parser.robot_parser import RobotParser
from example_parser.string_parser import StringParser
from search.MCTS.mcts import MCTS
from search.a_star.a_star import AStar
from search.abstract_search import SearchAlgorithm
from search.metropolis_hastings.metropolis import MetropolisHasting
from search.search_result import SearchResult
from search.brute.brute import Brute


@dataclass
class Runner:
    """Runner for running a program synthesizer for a given domain specific language NOT FOR a meta-synthesizer"""
    dsl: DomainSpecificLanguage = StandardDomainSpecificLanguage("robot")
    search_method: SearchAlgorithm = Brute(10)
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

                results = [r.get() for r in results]


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
        print(average_success_percentage)
        print(average_execution_time)
        return {
            "average_success": average_success_percentage,
            "average_execution": average_execution_time,
            "completely_succesful_percentage": percentage_of_completely_successful_programs,
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

        for file in os.listdir(directory):
            test_cases.append(parser.parse_file(file))

        return test_cases

    def run_single_test_case(self, test_case: TestCase):
        start_time = time.time()

        # # find program that satisfies training_examples
        search_result: SearchResult = self.search_method.run(test_case.training_examples, self.dsl)

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


"""
Example for running a test with the runner:
"""
if __name__ == '__main__':
    data = Runner().run()
    print(data)
