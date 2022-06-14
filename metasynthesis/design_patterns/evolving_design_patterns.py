import copy
import random
from typing import List, Callable, Tuple, Iterable

import numpy as np

from common.program_synthesis.dsl import DomainSpecificLanguage, StandardDomainSpecificLanguage
from common.tokens import robot_tokens, pixel_tokens, string_tokens
from solver.runner.runner import Runner
from solver.runner.algorithms import dicts
from common.tokens.abstract_tokens import BoolToken, InventedToken, TransToken, Token, PatternToken, \
    FunctionVariableToken
from metasynthesis.abstract_genetic import GeneticAlgorithm, Genome, Population

Genome = List[PatternToken]
Population = List[Genome]

fitness_dict = {}


class EvolvingDesignPatterns(GeneticAlgorithm):
    def __init__(self, fitness_limit: int, generation_limit: int, crossover_probability: float,
                 mutation_probability: float, generation_size: int, dsl: DomainSpecificLanguage,
                 search_algo: str, search_setting: str, search_mode: str, max_search_time: float):
        super().__init__(fitness_limit, generation_limit, crossover_probability, mutation_probability, generation_size)
        self.domain = dsl.domain_name
        print(dsl.domain_name)
        self.dsl = dsl
        self.trans_tokens = dsl.get_trans_tokens()
        self.bool_tokens = dsl.get_bool_tokens()
        self.search_algo = search_algo
        self.search_setting = search_setting
        self.search_mode = search_mode
        self.max_search_time = max_search_time

    def generate_function(self, body_length: int, param_occurrences: int) -> PatternToken:
        """This method creates a pattern"""
        if body_length <= 1 and body_length <= param_occurrences:
            raise ValueError()
        tokens: list[Token] = random.choices(self.trans_tokens, k=body_length - param_occurrences)
        for i in range(param_occurrences):
            tokens.insert(random.randint(0, len(tokens)), FunctionVariableToken("a"))
        return PatternToken(tokens, FunctionVariableToken("a"))

    def generate_genome(self, length: int) -> Genome:
        """This method creates a new genome of the specified length"""
        genome: Genome = []
        # max_body_length = 6
        # max_param_occurrences = 3
        # param_occurrences = random.randint(1, max_param_occurrences)
        # body_length = random.randint(max(2, param_occurrences), max_body_length)
        param_occurrences = 1
        body_length = 3

        for i in range(length):
            genome.append(self.generate_function(body_length, param_occurrences))
        return genome

    def generate_population(self) -> Population:
        """This method creates a population of new genomes"""
        max_number_of_patterns = 4
        population = []

        for i in range(self.generation_size):
            genome_length = random.randint(1, max_number_of_patterns)
            population.append(self.generate_genome(genome_length))

        return population

    def fitness(self, genome: Genome) -> float:
        """This method calculates the fitness of the specified genome"""
        # fitness of genome might've already been calculated
        if str(genome) in fitness_dict.keys():
            return fitness_dict[str(genome)]

        dsl = StandardDomainSpecificLanguage(self.dsl.domain_name)
        dsl.set_pattern_tokens(genome)

        runner = Runner(lib=dicts(0),
                        algo=self.search_algo,
                        setting=self.search_setting,
                        test_cases=self.search_mode,
                        time_limit_sec=self.max_search_time,
                        debug=True,
                        store=False,
                        suffix="",
                        dsl=dsl,
                        multi_thread=True,
                        )
        runner.run()

        # with ratios we ignore how many examples were there for a test case
        correct_ratios = []

        for result in runner.search_results.values():
            # the string domain uses test cases
            if self.domain == "string":
                correct_ratio = result["test_correct"] / result["test_total"]
            else:
                correct_ratio = result["train_correct"] / result["test_total"]

            correct_ratios.append(correct_ratio)

        mean_correct_ratios = np.mean(correct_ratios)

        fitness = mean_correct_ratios

        fitness_dict[str(genome)] = fitness

        return fitness

    def crossover(self, a: Genome, b: Genome) -> Tuple[Genome, Genome]:
        """This method applies the given crossover function with certain probability"""
        r = random.random()
        max_index = min(len(a), len(b))
        # indicates before which element to cut
        crossover_location = random.randrange(1, max_index)

        c, d = a, b

        if 0 <= r <= self.crossover_probability:
            c = a[:crossover_location] + b[crossover_location:]
            d = b[:crossover_location] + a[crossover_location:]

        return c, d

    def mutation(self, genome: Genome) -> Genome:
        """This method applies mutation to a genome with certain probability"""

        r = random.random()
        mutated_genome = genome

        if 0 <= r <= self.mutation_probability:
            mutated_genome = self.mutate_remove_token(genome)

        return mutated_genome

    def sort_population(self, population: Population) -> Population:
        """This method sorts the population based on the fitness function"""

        return sorted(population, key=lambda genome: self.fitness(genome), reverse=True)

    def genome_to_string(self, genome: Genome) -> str:
        """This method converts a given genome to a string"""

        raise NotImplementedError()

    def run_evolution(self):
        """This method runs the evolution process"""

        # initialize random population
        population = self.generate_population()
        generation_index = 0

        while generation_index < self.generation_limit:
            print("generation:", generation_index, "/", self.generation_limit)

            new_population = copy.deepcopy(population)
            # calculate fitness for genomes
            for genome in population:
                print(self.fitness(genome))

            # for ease of crossover, make the population even
            if len(population) % 2 != 0:
                print("population size odd!")

            # pick a pair of chromosomes for crossover
            selected_genomes = self.fitness_proportionate_selection(population)

            for i in range(0, len(population), 2):
                a = selected_genomes[i]
                b = selected_genomes[i+1]
                crossed_a, crossed_b = self.crossover(a, b)
                new_population[i] = crossed_a
                new_population[i+1] = crossed_b

            # mutation
            for i in range(len(new_population)):
                mutated_genome = self.mutation(copy.deepcopy(new_population[i]))
                new_population[i] = mutated_genome



    def fitness_proportionate_selection(self, population: Population):
        # roulette wheel sampling:
        # spins the wheel which consists of slices whose areas equal the genome fitness
        # it does so x times to make x/2 pairs for combination purposes
        weights = [self.fitness(x) for x in population]
        return random.choices(population, weights, k=len(population))

    def print_genome(self, genome: Genome):
        print("Genome: ", end="")
        for i, pattern in enumerate(genome):
            f = ", " if i < len(genome) - 1 else "\n"
            print(pattern, end=f)

    def print_population(self, population: Population):
        print("Population: ", end="[\n")
        for genome in population:
            print(end="\t")
            self.print_genome(genome)

        print("]")

    def mutate_replace_genome(self, genome: Genome):
        # should it be the same length or random?
        length = len(genome)
        return self.generate_genome(length)

    def mutate_add_token(self, genome: Genome):
        token = self.generate_function(3, 1)
        genome.append(token)
        return genome

    def mutate_remove_token(self, genome: Genome):
        random_token = random.choice(genome)
        print(random_token)
        genome.remove(random_token)
        return genome


if __name__ == "__main__":
    e = EvolvingDesignPatterns(1, 1, 1, 1, 6,
                               StandardDomainSpecificLanguage("robot"),
                               "Brute", "RG", "small", 0.55)
    a = e.generate_genome(4)
    b = e.generate_genome(3)

    p = e.generate_population()
    e.print_genome(a)
    # e.print_genome(b)
    # e.print_population(p)

    c, d = e.crossover(a, b)
    f = e.mutation(a)

    e.print_genome(f)
    # e.print_genome(c)
    # e.print_genome(d)
