# TESTING SERACH SEQUENCES
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Type

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

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
    allowed_searches: Dict[str, List[str]] = {"R": ["Brute", "AS", "LNS", "MH"], "S": ["GP", "Brute", "LNS", "MH", "AS"],
                                              "P": ["GP", "Brute", "LNS", "MH", "AS"]}

    # Initial populations are normally distributed, this dictionary contains respectively tuples with expectancy and std
    # TODO: test other distributions and var/std values
    initial_distribution_normal: Dict[str, Tuple[int, int]] = {"Brute": (4, 3), "AS": (10, 6), "MH": (300, 150),
                                                               "MCTS": (2000, 1000), "LNS": (1500, 1000), "GP": (2, 5)}

    initial_distribution_uniform: Dict[str, int] = {
        Brute: 6, MCTS: 4000, MetropolisHasting: 400, AStar: 15, LNS: 2800  # 10000
    }

    # Upper boundary for the execution time for specific search procedures, different for each domain
    initial_distribution_time: Dict[str, Dict[str, float]] = {"R": {"Brute": 0.5, "AS": 0.1, "MH": 0.05, "LNS": 0.3},
                                                              "S": {"Brute": 20,
                                                                    "AS": 12, "MH": 18,
                                                                    "LNS": 18, "GP": 20},
                                                              "P": {"Brute": 15,
                                                                    "AS": 6, "MH": 13,
                                                                    "LNS": 14, "GP": 20}}

    TIME_MAX = 1
    TOURN_SIZE = 2
    TESTS_SIZE = 100

    def __init__(self, fitness_limit: int, generation_limit: int, crossover_probability: float,
                 mutation_probability: float, generation_size: int, max_seq_size: int = 4, dist_type: str = "Time", setting: str =
                 "RE", print_generations: bool = False, test_size="small", plot: bool = False, write_generations: bool = False):
        super().__init__(fitness_limit, generation_limit, crossover_probability, mutation_probability, generation_size)

        self.test_size = test_size
        self.max_seq_size: int = max_seq_size

        self.curr_iteration: int = 1
        self.print_generations = print_generations

        # Dictionary containing genomes mapped to their average success percentage and execution time, so that they only need to be calculated once
        self.calculated_results: Dict[Tuple[Tuple[str, dict]], dict] = {}

        # List of mutations that are performed
        self.allowed_mutations: List[Type[MutationFunc]] = [self.replace_iteration_mutation, self.replace_search_mutation,
                                                            self.add_random_gene_mutation, self.delete_random_gene_mutation,
                                                            self.multiply_time_mutation]
        # Relative chance of specific mutations from allowed_mutations
        self.mutation_chances = [1, 1, 1, 1, 4]
        # Determine how the number of each search type are distributed
        self.dist_type = dist_type

        # Determine the exact type of the domain, including the parameters
        self.setting = setting

        # Used to determine the importance of metrics in the fitness function
        self.success_weight: float = 1
        self.time_weight: float = 0.2

        # dictionary mapping number of generation to list containing each gene along with its fitness value
        self.evolution_history: Dict[int, List[Tuple[Genome, float]]] = {}

        # List containing execution average execution times of successful searches in the current generation
        self.gen_times: List[float] = []
        self.avg_speed_per_gen: List[float] = []

        self.add_constant: float = 0.1
        self.subtract_constant: float = 0.1
        self.multiply_constant: float = 1.5
        self.divide_constant: float = 1.5

        # Choose whether to plot average fitness per generation or not
        self.plot = plot
        self.write_generations = write_generations

        if write_generations:
            self.start_time = datetime.now().strftime("%d-%m-%Y--%H-%M-%S")
            self.filename = f"{self.setting}-{self.mutation_probability}-p_m-{self.crossover_probability}-p_c-{self.start_time}"

    def generate_genome(self, length: int) -> Genome:
        new_genome: Genome = []

        for i in range(length):
            new_genome.append(self.generate_gene())

        return new_genome

    def generate_gene(self) -> tuple[str, float]:
        procedure: str = random.choices(self.allowed_searches[self.setting[0]])[0]
        time: float = self.generate_iterations(procedure, dist_type=self.dist_type)

        return procedure, time

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
            summed_time: float = self.calculated_results[tuple(genome)]['summed_time']
        # Or else run the synthesizer with a runner
        elif len(genome) != 0:
            print(f"Running search of {str(genome)}\n")
            runner: Runner = Runner(dicts(alg_sequence=genome), "CS", self.setting, self.test_size, 10, debug=False, store=False, multi_thread=True)
            success_ratio, average_time, _ = runner.run()
            average_success = success_ratio
            print("Finished running search\n")
            #     if self.dist_type == "Time":
            #         search: SearchAlgorithm = CombinedSearchTime(genome)
            #     else:
            #         search: SearchAlgorithm = CombinedSearch(0, genome)
            summed_time = np.sum([time for _, time in genome])
            # TODO: determine whether to take actual execution time or the summed up time from the list
            self.calculated_results[tuple(genome)] = {'average_success': average_success, 'average_time': average_time, 'summed_time': summed_time}

        # Assign 0 as a fitness if the genome is empty
        else:
            self.calculated_results[tuple(genome)] = {'average_success': 0, 'average_time': 0, 'summed_time': 0}
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
        self.save_generation(curr_population)

        if self.print_generations:
            print(self.population_to_string(curr_population))

        if self.write_generations:

            f = open(f"{self.filename}.txt", "a")
            f.write(self.population_to_string(curr_population))
            f.close()

        for gen in range(self.generation_limit - 1):
            self.curr_iteration += 1
            curr_population = self.generate_new_generation(curr_population)
            if self.print_generations:
                print(self.population_to_string(curr_population))
            if self.write_generations:
                f = open(f"{self.filename}.txt", "a")
                f.write(self.population_to_string(curr_population))
                f.close()
            self.calculate_population_fitness(curr_population)

        if self.plot:
            self.plot_generations_fitness()
            self.plot_generations_speed()

        return self.get_fittest(curr_population)

    def calculate_population_fitness(self, population: Population):
        for ind in population:
            self.fitness(ind)

    def generate_new_generation(self, old_population: Population):
        """
        Generate new population using crossover and mutation operators.
        """
        new_population: Population = []
        self.gen_times = []
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

        # TODO: figure out why this decreases running speed
        # if self.curr_iteration == self.generation_limit:
        #     new_population = self.coalesce_population(new_population)
        self.save_generation(new_population)

        return new_population

    def save_generation(self, population: Population):
        """
        Evaluates and saves the members of the given population along with their fitness.
        """
        self.evolution_history[self.curr_iteration] = []
        exec_times = []
        for ind in population:
            self.evolution_history[self.curr_iteration].append((ind, self.fitness(ind)))
            exec_times.append(self.calculated_results[tuple(ind)]['average_time'])
        self.avg_speed_per_gen.append(np.mean(exec_times))

    def plot_generations_fitness(self):
        """
        Plots the average fitness of each generation in evolution history.
        """

        avg_fitness: List = []
        for generation in self.evolution_history.values():
            fit_mean = np.mean([fitness for genotype, fitness in generation])
            avg_fitness.append(fit_mean)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.xlabel("Generation")
        plt.ylabel("Average fitness")
        plt.title(f"Fitness over generations\n{self.setting}, {self.mutation_probability} $p_m$, {self.crossover_probability} $p_c$")
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        x = np.arange(1, len(self.avg_speed_per_gen) + 1)
        plt.plot(x, avg_fitness)
        plt.savefig(f"fit-{self.filename}.png")
        plt.show()

    def plot_generations_speed(self):
        """
        Plots the average fitness of each generation in evolution history.
        """

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.xlabel("Generation")
        plt.ylabel("Time")
        plt.title(f"Average Execution Time of Successful Programs\n{self.setting}, {self.mutation_probability} $p_m$, {self.crossover_probability} "
                  f"$p_c$")
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        x = np.arange(1, len(self.avg_speed_per_gen) + 1)
        plt.plot(x, self.avg_speed_per_gen)
        plt.savefig(f"exec-{self.filename}.png")
        plt.show()

    def select_tournament(self, population: Population, compete_size: int) -> Tuple[Genome, Genome]:
        """
        Runs tournament as a selection procedure. Selects two individuals to reproduce.
        """
        first_pool: Population = random.sample(population, k=compete_size)
        second_pool: Population = random.sample(population, k=compete_size)

        return self.get_fittest(first_pool), self.get_fittest(second_pool)

    def rank_based_roulette_wheel(self, population: Population) -> Tuple[Genome, Genome]:
        """
        Runs rank-based roulette wheel as a selection procedure. Selects two individuals to reproduce.
        """
        # TODO: finish this one
        pass

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

    def coalesce_population(self, population: Population) -> Population:
        new_pop: Population = []
        for genome in population:
            new_pop.append(self.coalesce_searches(genome))
        return new_pop

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

    def generate_iterations(self, search_type: str, dist_type: str = "Gauss") -> float:
        """
        Uses normal distribution to generate a random iteration count for a given search.
        """
        if dist_type == "Gauss":
            return max(1, round(random.gauss(self.initial_distribution_normal[search_type][0], self.initial_distribution_normal[search_type][1])))
        elif dist_type == "Uniform":
            return random.randrange(1, self.initial_distribution_uniform[search_type])
        elif dist_type == "Time":
            return random.uniform(0, self.initial_distribution_time[self.setting[0]][search_type])
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
        searches: List[str] = self.allowed_searches[self.setting[0]].copy()
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
        new_genome[point] = (genome[point][0], genome[point][1] * max(0, np.random.normal(1, 0.3)))

        return new_genome

    # def divide_time_mutation(self, genome: Genome):
    #     """
    #     Doubles the execution time of a random search procedure.
    #     """
    #     point: int = random.randrange(0, len(genome))
    #
    #     new_genome: Genome = genome.copy()
    #     new_genome[point] = (genome[point][0], genome[point][1] * self.multiply_constant)
    #
    #     return new_genome

    # def add_time_mutation(self, genome: Genome):
    #     """
    #     Doubles the execution time of a random search procedure.
    #     """
    #     point: int = random.randrange(0, len(genome))
    #
    #     new_genome: Genome = genome.copy()
    #     new_time = genome[point][1] + np.random.normal(0, 0.1)
    #     if new_time > 0:
    #         new_genome[point] = (genome[point][0], new_time)
    #
    #     return new_genome

    def delete_random_gene_mutation(self, genome: Genome):
        new_genome: Genome = genome.copy()
        if len(genome) > 1:
            point: int = random.randrange(0, len(genome))
            del (new_genome[point])
        return new_genome

    def add_random_gene_mutation(self, genome: Genome):
        new_genome: Genome = genome.copy()
        if len(genome) < self.max_seq_size:
            point: int = random.randrange(0, len(genome))

            new_genome.insert(point, self.generate_gene())
        return new_genome
    #
    # def subtract_time_mutation(self, genome: Genome):
    #     """
    #     Doubles the execution time of a random search procedure.
    #     """
    #     point: int = random.randrange(0, len(genome))
    #
    #     new_genome: Genome = genome.copy()
    #     new_genome[point] = (genome[point][0], genome[point][1] + self.subtract_constant)
    #
    #     return new_genome


if __name__ == "__main__":
    ss = SearchSynthesiser(fitness_limit=0, generation_limit=20, crossover_probability=0.8,
                           mutation_probability=0.2, generation_size=10, max_seq_size=6, dist_type="Time", print_generations=True,
                           setting="SO", test_size="param", plot=True, write_generations=True)
    ss.run_evolution()
