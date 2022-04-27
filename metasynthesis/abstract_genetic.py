from typing import List, Callable, Tuple

Genome = List[int]
Population = List[Genome]
PopulateFunc = Callable[[], Population]
FitnessFunc = Callable[[Genome], float]
SelectionFunc = Callable[[Population, FitnessFunc], Tuple[Genome, Genome]]
CrossoverFunc = Callable[[Genome, Genome], Tuple[Genome, Genome]]
MutationFunc = Callable[[Genome], Genome]


class GeneticAlgorithm:
    """Abstract interface for a genetic algorithm to synthesise elements used in program synthesis"""

    def __init__(self, fitness_limit: int, generation_limit: int):
        self.fitness_limit = fitness_limit
        self.generation_limit = generation_limit

    def generate_genome(self, length: int) -> Genome:
        """This method creates a new genome of the specified length"""

        raise NotImplementedError()

    def generate_population(self, size: int, genome_length: int) -> Population:
        """This method creates a population of new genomes of the specified size"""

        return [self.generate_genome(genome_length) for _ in range(size)]

    def fitness(self, genome: Genome) -> float:
        """This method calculates the fitness of the specified genome"""

        raise NotImplementedError()

    def single_point_crossover(self, a: Genome, b: Genome) -> Tuple[Genome, Genome]:
        """This method combines first half of genome a with second half of b and vice versa"""

        raise NotImplementedError()

    def n_point_crossover(self, a: Genome, b: Genome) -> Tuple[Genome, Genome]:
        """This method applies multiple point crossover"""

        raise NotImplementedError()

    def mutation(self, genome: Genome, probability: float) -> Genome:
        """This method applies mutation to a genome with a certain probability"""

        raise NotImplementedError()

    def selection_pair(self, population: Population) -> Population:
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

        raise NotImplementedError()
