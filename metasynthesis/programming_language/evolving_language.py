from itertools import chain
from random import randrange
from typing import List, Callable, Tuple, Iterable
from common.tokens.pixel_tokens import BoolTokens as PixelBool
from common.tokens.robot_tokens import BoolTokens as RobotBool
from common.tokens.string_tokens import BoolTokens as StringBool

from evaluation.experiment_procedure import test_performance_single_case_and_write_to_file
from example_parser.pixel_parser import PixelParser
from example_parser.robot_parser import RobotParser
from example_parser.string_parser import StringParser
from example_parser.parser import Parser
from common.experiment import Experiment

from search.brute.brute import Brute
from search.batch_run import BatchRun

from metasynthesis.abstract_genetic import GeneticAlgorithm

Genome = List
Population = List[Genome]
PopulateFunc = Callable[[], Population]
FitnessFunc = Callable[[Genome], float]
SelectionFunc = Callable[[Population, FitnessFunc], Tuple[Genome, Genome]]
CrossoverFunc = Callable[[Genome, Genome], Tuple[Genome, Genome]]
MutationFunc = Callable[[Genome], Genome]


class EvolvingLanguage(GeneticAlgorithm):
    """A genetic algorithm to synthesise a programing language for program synthesis"""

    def __init__(self, fitness_limit: int, generation_limit: int, crossover_probability: float,
                 mutation_probability: float, domain: str, bool_tokens: list, trans_tokens: list):
        super().__init__(fitness_limit, generation_limit, crossover_probability, mutation_probability)
        self.domain = domain
        self.bool_tokens = bool_tokens
        self.trans_tokens = trans_tokens

    def generate_genome(self, length: int) -> Genome:
        """This method creates a new genome of the specified length"""

        all_tokens = self.bool_tokens + self.trans_tokens

        genome = set()

        for i in range(0, length):
            random_index = randrange(0, len(all_tokens))
            random_token = all_tokens[random_index]
            genome.add(random_token)

        return sort_genome(list(genome))

    def generate_population(self, size: int) -> Population:
        """This method creates a population of new genomes"""

        all_tokens = self.bool_tokens + self.trans_tokens

        population = []

        for i in range(0, size):
            genome_length = randrange(1, len(all_tokens))
            population.append(self.generate_genome(genome_length))

        return population

    def fitness(self, genome: Genome) -> float:
        """This method calculates the fitness of the specified genome"""

        # need a combination of
        # 1) length of genome
        # 2) whether program that solves tasks was created
        # 3) number of iterations of searching

        genome_length = len(genome)

        # result = BatchRun(
        #     domain=self.domain,
        #     files=([], [], []),
        #     search_algorithm=[Brute, "brute"][0](10),
        #     print_results=False,
        #     multi_core=False
        # ).run()
        # from result we can get the relevant stats
        # print("AAAA", result)

        # ISSUE: we need a way to inject the DSL genome into BatchRun

        # alternative: use test cases to run experiments myself

        # ISSUE: we can inject a DSL, but runs very very slow
        filtered_tokens = filter_tokens(genome, self.domain)
        bool_tokens = filtered_tokens[0]
        trans_tokens = filtered_tokens[1]

        parser = _get_parser(self.domain)
        test_cases = parser.parse_specific_range([], [], [])

        for test_case in test_cases:
            succes, time = test_performance_single_case_and_write_to_file(test_case, trans_tokens, bool_tokens, Brute)
            print(succes, time)

        # ISSUE: this runs quite fast, but no way to inject custom DSL
        # experiment = Experiment("temp_experiment_name", self.domain, test_cases)
        # print(test_performance_single_experiment(experiment, Brute))

        return 0.5

    def single_point_crossover(self, a: Genome, b: Genome) -> Tuple[Genome, Genome]:
        """This method combines first half of genome a with second half of b and vice versa with certain probability"""

        raise NotImplementedError()

    def n_point_crossover(self, a: Genome, b: Genome) -> Tuple[Genome, Genome]:
        """This method applies multiple point crossover with certain probability"""

        raise NotImplementedError()

    def mutation(self, genome: Genome, func: MutationFunc) -> Genome:
        """This method applies mutation to a genome with certain probability"""

        raise NotImplementedError()

    def selection_pair(self, population: Population) -> Tuple[Genome, Genome]:
        """This method selects a pair of solutions to be the parent of a new solution"""

        raise NotImplementedError()

    def sort_population(self, population: Population, fitness_func: FitnessFunc) -> Population:
        """This method sorts the population based on the given fitness function"""

        return sorted(population, key=fitness_func, reverse=True)

    def genome_to_string(self, genome: Genome) -> str:
        """This method converts a given genome to a string"""

        raise NotImplementedError()

    def run_evolution(self) -> Tuple[Population, int]:
        """This method runs the evolution process"""

        iteration_count = 0

        population = self.generate_population(50)

        print(population)

        # for genome in population:
        #     print(self.fitness(genome))

        return self.generate_population(50), iteration_count


def sort_genome(genome: Genome) -> Genome:
    return sorted(genome, key=str, reverse=False)


def filter_tokens(genome: Genome, domain: str) -> Tuple[List, List]:
    filtered = ([], [])
    if domain == "pixel":
        for token in genome:
            if token in PixelBool:
                filtered[0].append(token)
            else:
                filtered[1].append(token)
    elif domain == "robot":
        for token in genome:
            if token in RobotBool:
                filtered[0].append(token)
            else:
                filtered[1].append(token)
    elif domain == "string":
        for token in genome:
            if token in StringBool:
                filtered[0].append(token)
            else:
                filtered[1].append(token)
    return filtered


def _get_parser(domain: str) -> Parser:
    if domain == "string":
        return StringParser()
    elif domain == "robot":
        return RobotParser()
    elif domain == "pixel":
        return PixelParser()
    else:
        raise Exception()