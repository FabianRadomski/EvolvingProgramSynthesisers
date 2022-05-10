# TESTING SERACH SEQUENCES
import random
from typing import Dict, List, Optional, Tuple, Type

from common.program_synthesis.runner import Runner
from metasynthesis.abstract_genetic import CrossoverFunc, GeneticAlgorithm, Genome, MutationFunc, Population
from search.a_star.a_star import AStar
from search.abstract_search import SearchAlgorithm
from search.brute.brute import Brute
from search.combined_search.combined_search import CombinedSearch
from search.MCTS.mcts import MCTS
from search.metropolis_hastings.metropolis import MetropolisHasting
from search.vlns.large_neighborhood_search.algorithms.remove_n_insert_n import RemoveNInsertN


class SearchSynthesiser(GeneticAlgorithm):
    # Search procedures considered while constructing a genome
    # TODO: add vanillaGP
    allowed_searches: List[Type[SearchAlgorithm]] = [Brute, MCTS, MetropolisHasting, AStar, RemoveNInsertN]

    # Initial populations are normally distributed, this dictionary contains respectively tuples with expectancy and std
    # TODO: test other distributions and var/std values
    initial_distribution: Dict[Type[SearchAlgorithm], Tuple[int, int]] = {Brute: (10, 3), AStar: (10, 3), MetropolisHasting: (100, 50),
                                                                          MCTS: (300, 10), RemoveNInsertN: (300, 100)}

    TOURN_SIZE = 2
    TESTS_SIZE = 50

    def __init__(self, fitness_limit: int, generation_limit: int, crossover_probability: float,
                 mutation_probability: float, generation_size: int, max_seq_size: int = 4, print_generations: bool = False):
        super().__init__(fitness_limit, generation_limit, crossover_probability, mutation_probability, generation_size)

        self.max_seq_size: int = max_seq_size

        self.curr_iteration: int = 0
        self.print_generations = print_generations

        # Dictionary containing genomes mapped to their fitness values, so that they only need to be calculated once
        self.calculated_fitness: Dict[Tuple[Tuple[Type[SearchAlgorithm], int]], float] = {}

    def generate_genome(self, length: int) -> Genome:
        new_genome: Genome = []

        for i in range(length):
            procedure: Type[SearchAlgorithm] = random.choices(self.allowed_searches)[0]
            iterations: int = self.generate_iterations(procedure)
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

        # Check if the fitness has already been calculated
        if tuple(genome) in self.calculated_fitness.keys():
            result: float = self.calculated_fitness[tuple(genome)]

        # Or else run the synthesizer with a runner
        elif len(genome) != 0:
            search: SearchAlgorithm = CombinedSearch(0, genome)
            runner: Runner = Runner(search_method=search, MAX_TEST_CASES=self.TESTS_SIZE)
            result: float = runner.run()['average_success']
            self.calculated_fitness[tuple(genome)] = result

        # Assign 0 as a fitness if the genome is empty
        else:
            result = 0
            self.calculated_fitness[tuple(genome)] = result

        return result

    def crossover(self, a: Genome, b: Genome, func: CrossoverFunc) -> Tuple[Genome, Genome]:
        return func(a, b)

    def mutation(self, genome: Genome, func: MutationFunc) -> Genome:
        return func(genome)

    def selection_pair(self, population: Population) -> Tuple[Genome, Genome]:
        pass

    def genome_to_string(self, genome: Genome) -> str:
        return str(genome)

    def population_to_string(self, population: Population) -> str:
        """
        Converts the given population and fitness of its members to a readable format.
        """
        pop_str: str = f"Generation: {self.curr_iteration}/{self.generation_limit}\n"
        for individual in population:
            pop_str += f"{self.genome_to_string(individual)} + with fitness + {str(self.fitness(individual))} + \n"

        return pop_str

    def run_evolution(self):
        curr_population: Population = self.generate_population()

        for gen in range(self.generation_limit):
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
                new_population.extend(list(self.crossover(first_genome, second_genome, self.one_point_crossover)))
            else:
                new_population.append(first_genome)
                new_population.append(second_genome)

        # Apply mutation operators
        for i, individual in enumerate(new_population):
            if random.random() < self.mutation_probability:
                # TODO: check other mutation methods
                new_population[i] = self.mutation(individual, self.replace_iteration_mutation)

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
            fitness: float = self.fitness(best_individual)
            if fitness >= best_fitness:
                best_individual = individual
                best_fitness = fitness
        return best_individual

    def generate_iterations(self, search_type: Type[SearchAlgorithm]):
        """
        Uses normal distribution to generate a random iteration count for a given search.
        """
        return round(random.gauss(self.initial_distribution[search_type][0], self.initial_distribution[search_type][1]))

    @staticmethod
    def one_point_crossover(a: Genome, b: Genome) -> Tuple[Genome, Genome]:
        """
        Simple one point mutation.
        """

        a_point: int = random.randrange(0, len(a) + 1)
        b_point: int = random.randrange(0, len(b) + 1)

        new_a: Genome = a[:a_point].copy()
        new_b: Genome = b[:b_point].copy()
        new_a.extend(b[b_point:].copy())
        new_b.extend(a[a_point:].copy())

        return new_a, new_b

    def replace_iteration_mutation(self, genome: Genome):
        """Changes the number of iteration for a random search procedures in the genome."""

        point = random.randrange(0, len(genome))
        new_a = genome.copy()
        new_a[point][1] = self.generate_iterations(new_a[point][0])

        return new_a

    # def seq_test(self):
    #     alg_seq = [(MetropolisHasting, 1000), (MCTS, 1000)]
    #     results = BatchRun(
    #         # Task domain
    #         domain="robot",
    #
    #         # Iterables for files name. Use [] to use all values.
    #         # This runs all files adhering to format "2-*-[0 -> 10]"
    #         # Thus, ([], [], []) runs all files for a domain.
    #         files=([], [], []),
    #
    #         # Search algorithm to be used
    #         search_algorithm=Brute(10),
    #
    #         # Prints out result when a test case is finished
    #         print_results=True,
    #
    #         # Use multi core processing
    #         multi_core=True,
    #
    #         # Use file_name= to append to a file whenever a run got terminated
    #         # Comment out argument to create new file.
    #         # file_name="VLNS-20211213-162128.txt"
    #     ).run_seq(alg_seq)
    #     # print(result)


if __name__ == "__main__":
    SearchSynthesiser(0, 5, 0.8, 0.01, 10, 4, True).run_evolution()
