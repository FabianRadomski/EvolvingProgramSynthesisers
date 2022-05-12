import random
from common.program_synthesis.dsl import DomainSpecificLanguage
from common.tokens.abstract_tokens import BoolToken, InventedToken, TransToken
from metasynthesis.abstract_genetic import GeneticAlgorithm, Genome, Population
from typing import List, Callable, Tuple, Iterable

Genome = DomainSpecificLanguage
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
        self.trans_tokens = dsl.get_trans_tokens
        self.bool_tokens = dsl.get_bool_tokens

    def generate_genome(self, length: int) -> Genome:
        """This method creates a new genome of the specified length"""
        all_tokens = self.bool_tokens + self.trans_tokens


        # rework to add invented tokens
        # invented_token = invent_token()
        selected_tokens = random.sample(all_tokens, length)

        bool_tokens = [x for x in selected_tokens if isinstance(x, BoolToken)]
        trans_tokens = [x for x in selected_tokens if isinstance(x, TransToken)]

        dsl = DomainSpecificLanguage(self.domain, bool_tokens, trans_tokens)
        
        return sort_genome(dsl)

    def generate_population(self) -> Population:
        """This method creates a population of new genomes"""

        raise NotImplementedError()

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


def sort_genome(genome: Genome) -> Genome:
    sorted_bool = sorted(genome.get_bool_tokens(), key=str, reverse=False)
    sorted_trans = sorted(genome.get_trans_tokens(), key=str, reverse=False)

    return DomainSpecificLanguage(genome.domain_name, sorted_bool, sorted_trans)
    


