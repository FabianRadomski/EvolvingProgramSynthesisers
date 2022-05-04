from itertools import chain
from random import randrange
from typing import List, Callable, Tuple, Iterable

from common.program_synthesis.dsl import DomainSpecificLanguage
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

Genome = DomainSpecificLanguage
Population = List[Genome]

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

        return dsl_genome

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

        # genome_length = len(genome.get_bool_tokens() + genome.get_trans_tokens())

        runner = Runner(self.dsl)

        avg_success_perc, avg_exec_time, perc_successful_programs, search_results = runner.run()

        # max values: avg_success = 100, avg_exec_time = ?, perc_successful_programs = 100, search_results = ?

        return avg_success_perc

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
            print(genome.get_bool_tokens() + genome.get_trans_tokens())
            # print(self.fitness(genome))

        sorted_population = self.sort_population(population)

        best_performer = sorted_population[0]
        print(best_performer.get_bool_tokens() + best_performer.get_trans_tokens())

        return self.generate_population(), iteration_count


def sort_genome(genome: Genome) -> Genome:
    sorted_trans = sorted(genome.get_trans_tokens(), key=str, reverse=False)
    sorted_bool = sorted(genome.get_bool_tokens(), key=str, reverse=False)

    return DomainSpecificLanguage(genome.domain_name, sorted_trans, sorted_bool)
