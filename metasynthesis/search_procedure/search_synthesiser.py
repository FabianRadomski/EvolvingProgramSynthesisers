# TESTING SERACH SEQUENCES
import random
from typing import Dict, List, Optional, Tuple, Type

import numpy as np

from metasynthesis.abstract_genetic import CrossoverFunc, GeneticAlgorithm, Genome, MutationFunc, Population
from solver.runner.algorithms import dicts
from solver.runner.runner import Runner
from solver.search.implementations.a_star import AStar
from solver.search.implementations.brute import Brute
from solver.search.implementations.large_neighborhood_search import LNS
from solver.search.implementations.MCTS.mcts import MCTS
from solver.search.implementations.metropolis import MetropolisHasting


class SearchSynthesiser(GeneticAlgorithm):
    # Search procedures considered while constructing a genome
    # TODO: add vanillaGP
    allowed_searches: List[str] = ["Brute", "AS", "LNS", "MH", "GP"]

    # Initial populations are normally distributed, this dictionary contains respectively tuples with expectancy and std
    # TODO: test other distributions and var/std values
    initial_distribution_normal: Dict[str, Tuple[int, int]] = {"Brute": (4, 3), "AS": (10, 6), "MH": (300, 150),
                                                               "MCTS": (2000, 1000), "LNS": (1500, 1000), "GP": (2, 5)}

    initial_distribution_uniform: Dict[str, int] = {
        Brute: 6, MCTS: 4000, MetropolisHasting: 400, AStar: 15, LNS: 2800  # 10000
    }

    initial_distribution_time: Dict[str, float] = {
        "Brute": 1, "AS": 1, "MH": 1,
        "MCTS": 1, "LNS": 1, "GP": 1  # 10000
    }

    TIME_MAX = 1
    TOURN_SIZE = 2
    TESTS_SIZE = 100

    def __init__(self, fitness_limit: int, generation_limit: int, crossover_probability: float,
                 mutation_probability: float, generation_size: int, max_seq_size: int = 4, dist_type: str = "Time", setting: str =
                 "RE", print_generations: bool = False, test_size="small"):
        super().__init__(fitness_limit, generation_limit, crossover_probability, mutation_probability, generation_size)

        self.test_size = test_size
        self.max_seq_size: int = max_seq_size

        self.curr_iteration: int = 0
        self.print_generations = print_generations

        # Dictionary containing genomes mapped to their average success percentage and execution time, so that they only need to be calculated once
        self.calculated_results: Dict[Tuple[Tuple[str, dict]], dict] = {}

        # List of mutations that are performed
        self.allowed_mutations: List[Type[MutationFunc]] = [self.replace_iteration_mutation, self.replace_search_mutation,
                                                            self.divide_time_mutation, self.multiply_time_mutation,
                                                            self.add_time_mutation, self.subtract_time_mutation]
        # Relative chance of specific mutations from allowed_mutations
        self.mutation_chances = [1, 1, 2, 2, 2, 2]
        # Determine how the number of each search type are distributed
        self.dist_type = dist_type

        # Determine the exact type of the domain, including the parameters
        self.setting = setting

        # Used to determine the importance of metrics in the fitness function
        self.success_weight = 0.5
        self.time_weight = 0.5

        self.add_constant = 0.1
        self.subtract_constant = 0.1
        self.multiply_constant = 1.5
        self.divide_constant = 1.5

    def generate_genome(self, length: int) -> Genome:
        new_genome: Genome = []

        for i in range(length):
            procedure: str = random.choices(self.allowed_searches)[0]
            iterations: int = self.generate_iterations(procedure, dist_type=self.dist_type)
            new_genome.append((procedure, iterations))

        return new_genome

    def generate_population(self) -> Population:
        new_population: Population = []

        for i in range(self.generation_size):
            new_population.append(self.generate_genome(random.randrange(1, self.max_seq_size + 1)))

        return new_population

    def fitness(self, genome: Genome) -> float:
        """
        Fitness is calculated based on the average percent of successes.
        """
        # TODO: check other fitness metrics.
        # Execution time vs number of iterations???

        # Check if the metrics have already been calculated
        if tuple(genome) in self.calculated_results.keys():
            average_success: float = self.calculated_results[tuple(genome)]['average_success']
            average_time: float = self.calculated_results[tuple(genome)]['average_time']
            summed_time: float = self.calculated_results[tuple(genome)]['average_time']
        # Or else run the synthesizer with a runner
        elif len(genome) != 0:
            print(f"Running search of {str(genome)}\n")
            runner: Runner = Runner(dicts(alg_sequence=genome), "CS", self.setting, self.test_size, 10, debug=False, store=False, multi_thread=True)
            success_ratio, average_time = runner.run()
            average_success = success_ratio
            print("Finished running search\n")
            #     if self.dist_type == "Time":
            #         search: SearchAlgorithm = CombinedSearchTime(genome)
            #     else:
            #         search: SearchAlgorithm = CombinedSearch(0, genome)
            summed_time = np.sum([time for _, time in genome])
            # TODO: determine whether to take actual execution time or the summed up time from the list
            self.calculated_results[tuple(genome)] = {'average_success': average_success, 'average_time': summed_time}

        # Assign 0 as a fitness if the genome is empty
        else:
            self.calculated_results[tuple(genome)] = {'average_success': 0, 'average_time': 0}
            return 0

        # Only consider time if the programs are fully correct
        if average_success == 1.0:
            fitness = self.success_weight * average_success + self.time_weight * (1 / summed_time)
        else:
            fitness = self.success_weight * average_success
        return fitness

    def crossover_func(self, a: Genome, b: Genome, func: CrossoverFunc) -> Tuple[Genome, Genome]:
        new_a, new_b = func(a, b)
        return new_a, new_b

    def mutation_func(self, genome: Genome, func: MutationFunc) -> Genome:
        return func(genome)

    def selection_pair(self, population: Population) -> Tuple[Genome, Genome]:
        pass

    def genome_to_string(self, genome: Genome) -> str:
        genome_str: str = "["
        for i, gene in enumerate(genome):
            genome_str += f"{str(gene[0])} {gene[1]}"
            if i != len(genome) - 1:
                genome_str += " -> "
        genome_str += "]"
        return genome_str

    def population_to_string(self, population: Population) -> str:
        """
        Converts the given population and fitness of its members to a readable format.
        """
        pop_str: str = f"Generation: {self.curr_iteration}/{self.generation_limit}\n"
        for individual in population:
            pop_str += f"{self.genome_to_string(individual)} with fitness {str(self.fitness(individual))}\n"

        return pop_str

    def run_evolution(self):
        curr_population: Population = self.generate_population()

        if self.print_generations:
            print(self.population_to_string(curr_population))

        for gen in range(self.generation_limit):
            self.curr_iteration += 1
            curr_population = self.generate_new_generation(curr_population)
            if self.print_generations:
                print(self.population_to_string(curr_population))

        return self.get_fittest(curr_population)

    def generate_new_generation(self, old_population: Population):
        """
        Generate new population using crossover and mutation operators.
        """
        new_population: Population = []

        # Generate new population through crossovers
        while len(new_population) < self.generation_size:

            first_genome: Genome
            second_genome: Genome

            first_genome, second_genome = self.select_tournament(old_population, self.TOURN_SIZE)
            if random.random() < self.crossover_probability:
                # TODO: check other crossover methods
                new_population.extend(list(self.crossover_func(first_genome, second_genome, self.two_point_crossover)))
            else:
                new_population.append(first_genome)
                new_population.append(second_genome)

        # Apply mutation operators
        for i, individual in enumerate(new_population):
            if random.random() < self.mutation_probability:
                # TODO: check other mutation methods
                new_population[i] = self.mutation_func(individual, random.choices(self.allowed_mutations, weights=self.mutation_chances)[0])

        return new_population

    def select_tournament(self, population: Population, compete_size: int) -> Tuple[Genome, Genome]:
        """
        Runs tournament as a selection procedure. Selects two individuals to reproduce.
        """
        first_pool: Population = random.sample(population, k=compete_size)
        second_pool: Population = random.sample(population, k=compete_size)

        return self.get_fittest(first_pool), self.get_fittest(second_pool)

    def get_fittest(self, pool: Population) -> Optional[Genome]:
        """
        Selects the fittest individual from the population(pool).
        """
        if pool == 0:
            return None
        best_fitness: float = 0
        best_individual: Genome = pool[0]
        for individual in pool:
            fitness: float = self.fitness(individual)
            if fitness >= best_fitness:
                best_individual = individual
                best_fitness = fitness
        return best_individual

    @staticmethod
    def coalesce_searches(sequence: Genome) -> Genome:
        """
        Merges identical consequent searches in the sequence.
        """
        merged_sequence: Genome = []
        curr_gene: (str, int) = (sequence[0][0], sequence[0][1])
        for i in range(len(sequence)):
            if i == 0:
                continue
            if sequence[i][0] == sequence[i - 1][0]:
                curr_gene = (curr_gene[0], curr_gene[1] + sequence[i][1])
            else:
                merged_sequence.append(curr_gene)
                curr_gene = (sequence[i][0], sequence[i][1])
        merged_sequence.append(curr_gene)

        return merged_sequence

    def generate_iterations(self, search_type: str, dist_type: str = "Gauss"):
        """
        Uses normal distribution to generate a random iteration count for a given search.
        """
        if dist_type == "Gauss":
            return max(1, round(random.gauss(self.initial_distribution_normal[search_type][0], self.initial_distribution_normal[search_type][1])))
        elif dist_type == "Uniform":
            return random.randrange(1, self.initial_distribution_uniform[search_type])
        elif dist_type == "Time":
            return random.uniform(0.05, self.initial_distribution_time[search_type])
        else:
            raise Exception("The chosen iteration distribution is not allowed. Choose either Gauss or Uniform!")

    @staticmethod
    def one_point_crossover(a: Genome, b: Genome) -> Tuple[Genome, Genome]:
        """
        Simple one point crossover.
        """

        a_point: int = random.randrange(1, len(a) + 1)
        b_point: int = random.randrange(1, len(b) + 1)

        new_a: Genome = a[:a_point].copy()
        new_b: Genome = b[:b_point].copy()
        new_a.extend(b[b_point:].copy())
        new_b.extend(a[a_point:].copy())

        return new_a, new_b

    @staticmethod
    def two_point_crossover(a: Genome, b: Genome) -> Tuple[Genome, Genome]:
        """
        Exchanges part of the genome of a random size.
        """
        exchange_size = random.randrange(1, min(len(a), len(b)) + 1)
        a_point: int = random.randrange(0, len(a) + 1 - exchange_size)
        b_point: int = random.randrange(0, len(b) + 1 - exchange_size)

        new_a: Genome = a[:a_point].copy()
        new_a.extend(b[b_point:b_point + exchange_size].copy())
        new_a.extend(a[a_point + exchange_size:].copy())

        new_b: Genome = b[:b_point].copy()
        new_b.extend(a[a_point:a_point + exchange_size].copy())
        new_b.extend(b[b_point + exchange_size:].copy())

        return new_a, new_b

    def replace_iteration_mutation(self, genome: Genome):
        """Changes the number of iteration for a random search procedures in the genome."""

        point: int = random.randrange(0, len(genome))
        new_genome: Genome = genome.copy()
        new_genome[point] = (genome[point][0], self.generate_iterations(new_genome[point][0], dist_type=self.dist_type))

        return new_genome

    def replace_search_mutation(self, genome: Genome):
        """
        Replaces one search procedure in the genome with a different random one.
        """
        point: int = random.randrange(0, len(genome))

        removed_search: str = genome[point][0]
        searches: List[str] = self.allowed_searches.copy()
        searches.remove(removed_search)

        new_genome: Genome = genome.copy()
        new_search: str = random.choice(searches)
        new_genome[point] = (new_search, genome[point][1])

        return new_genome

    def multiply_time_mutation(self, genome: Genome):
        """
        Decreases the execution time of a random search procedure by half.
        """
        point: int = random.randrange(0, len(genome))

        new_genome: Genome = genome.copy()
        new_genome[point] = (genome[point][0], genome[point][1] / self.divide_constant)

        return new_genome

    def divide_time_mutation(self, genome: Genome):
        """
        Doubles the execution time of a random search procedure.
        """
        point: int = random.randrange(0, len(genome))

        new_genome: Genome = genome.copy()
        new_genome[point] = (genome[point][0], genome[point][1] * self.multiply_constant)

        return new_genome

    def add_time_mutation(self, genome: Genome):
        """
        Doubles the execution time of a random search procedure.
        """
        point: int = random.randrange(0, len(genome))

        new_genome: Genome = genome.copy()
        new_genome[point] = (genome[point][0], genome[point][1] + self.add_constant)

        return new_genome

    def subtract_time_mutation(self, genome: Genome):
        """
        Doubles the execution time of a random search procedure.
        """
        point: int = random.randrange(0, len(genome))

        new_genome: Genome = genome.copy()
        new_genome[point] = (genome[point][0], genome[point][1] + self.subtract_constant)

        return new_genome


if __name__ == "__main__":
    SearchSynthesiser(fitness_limit=0, generation_limit=50, crossover_probability=0.8,
                      mutation_probability=0.05, generation_size=20, max_seq_size=4, dist_type="Time", print_generations=True,
                      setting= "PG").run_evolution()
