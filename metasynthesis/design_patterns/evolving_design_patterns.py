import copy
import random
from typing import List, Callable, Tuple, Iterable

import numpy as np

from common.program_synthesis.dsl import DomainSpecificLanguage, StandardDomainSpecificLanguage
from solver.runner.runner import Runner
from solver.runner.algorithms import dicts
from common.tokens.abstract_tokens import BoolToken, InventedToken, TransToken, Token, PatternToken, \
    FunctionVariableToken
from metasynthesis.abstract_genetic import GeneticAlgorithm, Genome, Population

Genome = List[PatternToken]
Population = List[Genome]

fitness_dict = {}


class EvolvingDesignPatterns(GeneticAlgorithm):
    def __init__(self, fitness_limit: int = 1, generation_limit: int = 30,
                 crossover_probability: float = 0.60, mutation_probability: float = 0.005,
                 generation_size: int = 50, elite_genomes: int = 5,
                 dsl: DomainSpecificLanguage = StandardDomainSpecificLanguage("robot"),
                 algo: str = "AS", setting: str = "RO", test_cases: str = "debug",
                 time_limit_sec: float = 1,
                 debug: bool = False, multi_thread: bool = True):
        super().__init__(fitness_limit, generation_limit, crossover_probability, mutation_probability, generation_size)
        self.domain = dsl.domain_name
        self.dsl = dsl
        self.elite_genomes = elite_genomes
        self.trans_tokens = dsl.get_trans_tokens()
        self.bool_tokens = dsl.get_bool_tokens()
        self.algo = algo
        self.setting = setting
        self.test_cases = test_cases
        self.time_limit_sec = time_limit_sec
        self.debug = debug
        self.multi_thread = multi_thread
        self.default_cost = 0.0

    def generate_function(self, body_length: int, param_occurrences: int) -> PatternToken:
        """This method creates a pattern"""
        if body_length <= 1 and body_length < param_occurrences:
            raise ValueError()
        tokens: list[Token] = random.choices(self.trans_tokens, k=body_length - param_occurrences)
        for i in range(param_occurrences):
            tokens.insert(random.randint(0, len(tokens)), FunctionVariableToken("a"))
        return PatternToken(tokens, FunctionVariableToken("a"))

    def generate_genome(self, length: int) -> Genome:
        """This method creates a new genome of the specified length"""
        genome: Genome = []
        max_body_length = 6
        max_param_occurrences = 6
        # param_occurrences = random.randint(1, max_param_occurrences)
        # body_length = random.randint(max(2, param_occurrences), max_body_length)
        # param_occurrences = 1
        # body_length = 3

        for i in range(length):
            param_occurrences = random.randint(1, max_param_occurrences)
            body_length = random.randint(max(2, param_occurrences), max_body_length)
            genome.append(self.generate_function(body_length, param_occurrences))
        return genome

    def generate_population(self) -> Population:
        """This method creates a population of new genomes"""
        min_number_of_patterns = 2
        max_number_of_patterns = 6
        population = []

        for i in range(self.generation_size):
            genome_length = random.randint(min_number_of_patterns, max_number_of_patterns)
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
                        algo=self.algo,
                        setting=self.setting,
                        test_cases=self.test_cases,
                        time_limit_sec=self.time_limit_sec,
                        debug=self.debug,
                        store=False,
                        suffix="",
                        dsl=dsl,
                        multi_thread=self.multi_thread,
                        )
        runner.run()

        profit_avg, time_avg, correct_avg = get_averages_from_search_results(runner.search_results, self.domain)

        fitness = correct_avg * (1 / time_avg) if time_avg > 0 else 0

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
            mutated_genome = self.mutate_replace_genome(genome)

        return mutated_genome

    def sort_population(self, population: Population) -> Population:
        """This method sorts the population based on the fitness function"""

        return sorted(population, key=lambda genome: self.fitness(genome), reverse=True)

    def genome_to_string(self, genome: Genome) -> str:
        """This method converts a given genome to a string"""

        raise NotImplementedError()

    def run_evolution(self):
        """This method runs the evolution process"""
        fitness_dict.clear()

        # print("default fitness: ", self.fitness([]))

        # initialize random population
        population = self.generate_population()

        max_fitnesses_per_generation = []
        avg_fitnesses_per_generation = []
        best_pattern = []
        generation_index = 0

        while generation_index < self.generation_limit:
            print("generation:", generation_index+1, "/", self.generation_limit)

            new_population = copy.deepcopy(population)

            for genome in new_population:
                fitness = self.fitness(genome)

            # picking elite genomes
            sorted_population = sorted(new_population, key=lambda x: self.fitness(x), reverse=True)
            elites = sorted_population[0:self.elite_genomes]
            # they go straight into the new population
            new_population[0:self.elite_genomes] = elites

            # for ease of crossover, make sure the population even
            if len(new_population) % 2 != 0:
                print("population size odd!")

            # pick a number of chromosomes for crossover
            selected_genomes = self.fitness_proportionate_selection(population, self.generation_size)
            non_elite = self.generation_size - self.elite_genomes
            for i in range(0, non_elite, 2):
                # pick a pair of genomes and cross them over
                a = selected_genomes[i]
                b = selected_genomes[i + 1]
                crossed_a, crossed_b = self.crossover(a, b)
                new_population[self.elite_genomes + i] = crossed_a
                # if the non-elite genomes are odd, we need to get rid of 1 genome in last pair
                if i + 1 != non_elite:
                    new_population[self.elite_genomes + i + 1] = crossed_b

            # mutation
            for i in range(self.elite_genomes, self.generation_size):
                mutated_genome = self.mutation(copy.deepcopy(new_population[i]))
                new_population[i] = mutated_genome

            # collect stats
            fitnesses = []
            for genome in population:
                fitnesses.append(self.fitness(genome))

            max_fitness = max(fitnesses)
            max_fitnesses_per_generation.append(max_fitness)
            best_pattern = population[fitnesses.index(max_fitness)]
            avg_fitness = np.mean(fitnesses)
            avg_fitnesses_per_generation.append(avg_fitness)

            generation_index += 1
            population = new_population

        print("best pattern was found in generation: ", max_fitnesses_per_generation.index(max_fitness)+1)

        print("Best evolved pattern:")
        self.evaluation(best_pattern)
        print("No patterns:")
        self.evaluation([])

        return {
            "max_fitnesses": max_fitnesses_per_generation,
            "avg_fitnesses": avg_fitnesses_per_generation,
            "population": population,
            "best_pattern": best_pattern
        }


    def evaluation(self, genome: Genome):
        dsl = StandardDomainSpecificLanguage(self.domain)
        dsl.set_pattern_tokens(genome)

        runner = Runner(lib=dicts(0),
                        algo=self.algo,
                        setting=self.setting,
                        test_cases=self.test_cases,
                        time_limit_sec=self.time_limit_sec,
                        debug=False,
                        store=False,
                        suffix="",
                        dsl=dsl,
                        multi_thread=self.multi_thread,
                        )
        runner.run()

        profit_avg, time_avg, correct_avg = get_averages_from_search_results(runner.search_results, self.domain)
        fitness = correct_avg * (1 / time_avg) if time_avg > 0 else 0
        print("Profit average: ", profit_avg)
        print("Correct average: ", correct_avg)
        print("Time average: ", time_avg)
        print("Overall fitness: ", fitness)
        print_genome(genome)

    def fitness_proportionate_selection(self, population: Population, size: int = None):
        # roulette wheel sampling:
        # spins the wheel which consists of slices whose areas equal the genome fitness
        # it does so x times to make x/2 pairs for combination purposes
        # using tiny value for weight instead of 0 prevents breaking in case all fitness values are 0
        weights = [self.fitness(x) if self.fitness(x) > 0 else np.nextafter(0, 1) for x in population]
        if size is None:
            size = len(population)
        return random.choices(population, weights, k=size)

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


def get_averages_from_search_results(search_results: dict, domain: str) -> Tuple[float, float]:
    profit_total = 0
    time_total = 0
    correct_total = 0

    results = search_results.values()

    for result in results:
        search_time = result["search_time"]
        test_total = result["test_total"]

        if domain == "string":
            cost = result["test_cost"]
            profit = 1 - (cost if cost <= 1 else 1)
            correct = result["test_correct"]/test_total
        else:
            cost = result["train_cost"]
            profit = 1 - (cost if cost <= 1 else 1)
            correct = result["train_correct"]/test_total

        profit_total += profit
        time_total += search_time
        correct_total += correct

    profit_avg = profit_total / len(results)
    time_avg = time_total / len(results)
    correct_avg = correct_total / len(results)
    return profit_avg, time_avg, correct_avg


def print_genome(genome: Genome):
    print("Genome: ", end="")
    for i, pattern in enumerate(genome):
        f = ", " if i < len(genome) - 1 else "\n"
        print(pattern, end=f)


def print_population(population: Population):
    print("Population: ", end="[\n")
    for genome in population:
        print(end="\t")
        print_genome(genome)
    print("]")
