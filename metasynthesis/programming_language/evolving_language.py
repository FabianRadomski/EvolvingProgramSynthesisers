from itertools import chain
from random import randrange
from typing import List, Callable, Tuple, Iterable

from common.program_synthesis.dsl import DomainSpecificLanguage, StandardDomainSpecificLanguage
from common.program_synthesis.objective import ObjectiveFun
from common.program_synthesis.runner import Runner
from common.tokens.pixel_tokens import BoolTokens as PixelBool
from common.tokens.robot_tokens import BoolTokens as RobotBool
from common.tokens.string_tokens import BoolTokens as StringBool

from evaluation.experiment_procedure import test_performance_single_case_and_write_to_file
from example_parser.pixel_parser import PixelParser
from example_parser.robot_parser import RobotParser
from example_parser.string_parser import StringParser
from example_parser.parser import Parser

from metasynthesis.abstract_genetic import GeneticAlgorithm
from metasynthesis.abstract_genetic import FitnessFunc
from metasynthesis.abstract_genetic import CrossoverFunc
from metasynthesis.abstract_genetic import MutationFunc
from search.brute.brute import Brute

Genome = DomainSpecificLanguage
Population = List[Genome]

genome_fitness_values = {}


class EvolvingLanguage(GeneticAlgorithm):
    """A genetic algorithm to synthesise a programing language for program synthesis"""

    def __init__(self, fitness_limit: int, generation_limit: int, crossover_probability: float,
                 mutation_probability: float, generation_size: int, dsl: DomainSpecificLanguage):
        super().__init__(fitness_limit, generation_limit, crossover_probability, mutation_probability, generation_size)
        self.domain = dsl.domain_name
        self.dsl = dsl
        self.bool_tokens = dsl.get_bool_tokens()
        self.trans_tokens = dsl.get_trans_tokens()

    def generate_genome(self, length: int) -> Genome:
        """This method creates a new genome of the specified length"""

        max_dsl_length = len(self.bool_tokens + self.trans_tokens)
        bool_length = round((length / max_dsl_length) * len(self.bool_tokens))
        trans_length = round((length / max_dsl_length) * len(self.trans_tokens))

        bool_chromosomes = set()
        for i in range(0, bool_length):
            random_index = randrange(0, len(self.bool_tokens))
            random_bool_token = self.bool_tokens[random_index]
            bool_chromosomes.add(random_bool_token)

        trans_chromosomes = set()
        for i in range(0, trans_length):
            random_index = randrange(0, len(self.trans_tokens))
            random_trans_token = self.trans_tokens[random_index]
            trans_chromosomes.add(random_trans_token)

        dsl_genome = DomainSpecificLanguage(self.domain, list(bool_chromosomes), list(trans_chromosomes))

        return sort_genome(dsl_genome)

    def generate_population(self) -> Population:
        """This method creates a population of new genomes"""

        max_dsl_length = len(self.bool_tokens + self.trans_tokens)

        population = []

        for i in range(0, self.generation_size):
            genome_length = randrange(1, max_dsl_length)
            population.append(self.generate_genome(genome_length))

        return population

    def fitness(self, genome: Genome) -> float:
        """This method calculates the fitness of the specified genome"""

        # need a combination of
        # 1) length of genome
        # 2) whether program that solves tasks was created
        # 3) number of iterations of searching

        if genome.to_string() in genome_fitness_values.keys():
            print("ALREADY IN", genome.to_string())
            return genome_fitness_values[genome.to_string()]

        # genome = StandardDomainSpecificLanguage("robot")
        runner = Runner(dsl=genome,
                        search_method=Brute(1, ObjectiveFun(self.domain).fun),
                        max_test_cases=200)
        results = runner.run()

        dsl_length = len(genome.get_bool_tokens() + genome.get_trans_tokens())
        inverse_dsl_length = 1 / dsl_length

        avg_exec_time = results["average_execution"]
        inverse_avg_exec_time = 1 / (avg_exec_time + 0.000000001)

        success_percentage = results["average_success"]
        success_percentage_scaled = success_percentage / 100

        print(inverse_dsl_length, inverse_avg_exec_time, success_percentage_scaled)

        fitness_value = inverse_dsl_length * inverse_avg_exec_time * success_percentage_scaled

        genome_fitness_values[genome.to_string()] = fitness_value

        return fitness_value

    def crossover(self, a: Genome, b: Genome, func: CrossoverFunc) -> Tuple[Genome, Genome]:
        """This method applies the given crossover function with certain probability"""

        raise NotImplementedError()

    def mutation(self, genome: Genome, func: MutationFunc) -> Genome:
        """This method applies mutation to a genome with certain probability"""

        raise NotImplementedError()

    def selection_pair(self, population: Population) -> Tuple[Genome, Genome]:
        """This method selects a pair of solutions to be the parent of a new solution"""

        raise NotImplementedError()

    def sort_population(self, population: Population) -> Population:
        """This method sorts the population based on the given fitness function"""

        return sorted(population, key=lambda genome: self.fitness(genome), reverse=True)

    def genome_to_string(self, genome: Genome) -> str:
        """This method converts a given genome to a string"""

        raise NotImplementedError()

    def run_evolution(self):
        """This method runs the evolution process"""

        iteration_count = 0

        population = self.generate_population()

        for genome in population:
            print(genome.to_string())
            print(self.fitness(genome))

        best_percentage = select_best_percentage(population=population, percentage=50, size=self.generation_size)

        

        sorted_population = sorted(population, key=lambda x: get_fitness(x), reverse=True)
        best_performer = sorted_population[0]
        print(best_performer.to_string())

        return self.generate_population(), iteration_count


def sort_genome(genome: Genome) -> Genome:
    sorted_trans = sorted(genome.get_trans_tokens(), key=str, reverse=False)
    sorted_bool = sorted(genome.get_bool_tokens(), key=str, reverse=False)

    return DomainSpecificLanguage(genome.domain_name, sorted_bool, sorted_trans)


def genome_length(genome: Genome) -> int:
    return len(genome.get_bool_tokens() + genome.get_trans_tokens())


def get_fitness(genome: Genome):
    return genome_fitness_values[genome.to_string()]


def select_best_percentage(population: Population, percentage: float, size: int) -> Population:
    sorted_population = sorted(population, key=lambda x: get_fitness(x), reverse=True)

    max_to_get = round((percentage / 100) * size)
    print(max_to_get)
    return population[:max_to_get]
