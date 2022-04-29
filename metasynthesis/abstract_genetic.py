from typing import List, Callable, Tuple

Genome = List
Population = List[Genome]
PopulateFunc = Callable[[], Population]
FitnessFunc = Callable[[Genome], float]
SelectionFunc = Callable[[Population, FitnessFunc], Tuple[Genome, Genome]]
CrossoverFunc = Callable[[Genome, Genome], Tuple[Genome, Genome]]
MutationFunc = Callable[[Genome], Genome]


class GeneticAlgorithm:
    """Abstract interface for a genetic algorithm to synthesise elements used in program synthesis"""

    def __init__(self, fitness_limit: int, generation_limit: int, crossover_probability: float,
                 mutation_probability: float, generation_size: int):
        self.fitness_limit = fitness_limit
        self.generation_limit = generation_limit
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.generation_size = generation_size

    def generate_genome(self, length: int) -> Genome:
        """This method creates a new genome of the specified length"""

        raise NotImplementedError()

    def generate_population(self) -> Population:
        """This method creates a population of new genomes"""

        raise NotImplementedError()

    def fitness(self, genome: Genome) -> float:
        """This method calculates the fitness of the specified genome"""

        raise NotImplementedError()

    def crossover(self, a: Genome, b: Genome, func: CrossoverFunc) -> Tuple[Genome, Genome]:
        """This method applies the given crossover function with certain probability"""

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

    def run_evolution(self):
        """This method runs the evolution process"""

        raise NotImplementedError()
