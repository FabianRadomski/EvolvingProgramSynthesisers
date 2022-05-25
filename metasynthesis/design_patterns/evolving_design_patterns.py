import random
from common.program_synthesis.dsl import DomainSpecificLanguage
from common.tokens.abstract_tokens import BoolToken, InventedToken, TransToken, Token, FunctionDefinitionToken, FunctionVariableToken, PatternToken
from metasynthesis.abstract_genetic import GeneticAlgorithm, Genome, Population
from typing import List, Callable, Tuple, Iterable

Genome = List[PatternToken]
Population = List[Genome]


class EvolvingDesignPatterns(GeneticAlgorithm):
    def __init__(self, fitness_limit: int, generation_limit: int, crossover_probability: float,
                 mutation_probability: float, generation_size: int, dsl: DomainSpecificLanguage):
        self.fitness_limit = fitness_limit
        self.generation_limit = generation_limit
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.generation_size = generation_size
        self.dsl = dsl
        self.trans_tokens = dsl.get_trans_tokens()
        self.bool_tokens = dsl.get_bool_tokens()

    def generate_function(self, body_length: int, param_occurrences: int) -> FunctionDefinitionToken:
        """This method creates a function definition for a pattern"""
        if body_length <= 1 and body_length <= param_occurrences:
            raise ValueError()
        tokens: list[Token] = random.choices(self.trans_tokens, k=body_length - param_occurrences)
        for i in range(param_occurrences):
            tokens.insert(random.randint(0, len(tokens)), FunctionVariableToken("a"))
        return FunctionDefinitionToken(tokens, FunctionVariableToken("a"))

    def generate_genome(self, length: int) -> Genome:
        """This method creates a new genome of the specified length"""
        genome = []
        # max_body_length = 6
        # max_param_occurrences = 3
        # param_occurrences = random.randint(1, max_param_occurrences)
        # body_length = random.randint(max(2, param_occurrences), max_body_length)
        body_length = 3
        param_occurrences = 1
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

        raise NotImplementedError()

    def crossover(self, a: Genome, b: Genome) -> Tuple[Genome, Genome]:
        """This method applies the given crossover function with certain probability"""

        raise NotImplementedError()

    def mutation(self, genome: Genome) -> Genome:
        """This method applies mutation to a genome with certain probability"""

        raise NotImplementedError()

    def sort_population(self, population: Population) -> Population:
        """This method sorts the population based on the fitness function"""

        return sorted(population, key=lambda genome: self.fitness(genome), reverse=True)

    def genome_to_string(self, genome: Genome) -> str:
        """This method converts a given genome to a string"""

        raise NotImplementedError()

    def run_evolution(self):
        """This method runs the evolution process"""

        raise NotImplementedError()
