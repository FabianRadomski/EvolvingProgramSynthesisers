from random import randrange
from typing import List, Callable, Tuple

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
                 mutation_probability: float, domain: str, all_tokens: list):
        super().__init__(fitness_limit, generation_limit, crossover_probability, mutation_probability)
        self.domain = domain
        self.all_tokens = all_tokens

    def generate_genome(self, length: int) -> Genome:
        """This method creates a new genome of the specified length"""

        genome = []

        for i in range(0, length):
            random_index = randrange(0, len(self.all_tokens))
            random_token = self.all_tokens[random_index]
            genome.append(random_token)

        return sort_genome(genome)

    def generate_population(self, size: int) -> Population:
        """This method creates a population of new genomes"""

        population = []

        for i in range(0, size):
            genome_length = randrange(1, len(self.all_tokens))
            population.append(self.generate_genome(genome_length))

        return population

    def fitness(self, genome: Genome) -> float:
        """This method calculates the fitness of the specified genome"""

        raise NotImplementedError()

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

        print(self.generate_population(100))

        return self.generate_population(100), 1


def sort_genome(genome: Genome):
    return sorted(genome, key=str, reverse=False)

