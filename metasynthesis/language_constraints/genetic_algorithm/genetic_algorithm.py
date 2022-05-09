import math
from copy import deepcopy
import numpy as np
import random
from typing import Tuple, List

from common.tokens.abstract_tokens import Token
from common.program_synthesis.runner import Runner
from common.program_synthesis.dsl import DomainSpecificLanguage, robot_tokens, pixel_tokens, string_tokens
from metasynthesis.abstract_genetic import GeneticAlgorithm, Population, Genome, MutationFunc
from metasynthesis.language_constraints.constraints.Constraints import AbstractConstraint


class ConstraintGeneticAlgorithm(GeneticAlgorithm):

    def __init__(self, fitness_limit: float,
                 generation_limit: int,
                 mutation_probability: float,
                 constraints: List[Constraints],
                 population_size: int = 20):
        super().__init__(fitness_limit, generation_limit, 0, mutation_probability)
        self.constraints = constraints
        self.max_genome_value = list(map(lambda c: c.get_values(), constraints))
        self.population_size = population_size

    def generate_genome(self, length: int = -1) -> Genome:
        genome = map(lambda x: math.floor(random.random() * x), self.max_genome_value)
        return list(genome)

    def generate_population(self, size: int = -1) -> Population:
        return [self.generate_genome() for _ in range(self.population_size)]

    def _derive_domain_func(self, genome: Genome, bool_tokens: List[Token], trans_tokens: List[Token]):
        # generate constraints from genome
        constraints = list(map(lambda c, v: c.deepcopy().set_value(v), zip(self.constraints, genome)))

        def constraint_builder_func(constraints: List[AbstractConstraint], bool_tokens: List[Token],
                                    trans_tokens: List[Token]):
            def constraint_func(sequence: List[Token]):
                bt = bool_tokens.copy()
                tt = trans_tokens.copy()
                for constraint in constraints:
                    for active_constraint in constraint.constraint(sequence):
                        bt.remove(active_constraint)
                        tt.remove(active_constraint)

                return bt, tt

            return constraint_func

        return constraint_builder_func(constraints, bool_tokens, trans_tokens)

    def _create_dsl(self, genome: Genome, domain):
        if domain == 'robot':
            module = robot_tokens
        elif domain == 'pixel':
            module = pixel_tokens
        else:
            module = string_tokens

        bt = module.BoolTokens
        tt = module.TransTokens
        func = self._derive_domain_func(genome, bt, tt)

        return DomainSpecificLanguage(domain, bt, tt, True, func)

    def fitness(self, genome: Genome) -> float:
        dsl = self._create_dsl(genome, 'robot')
        runner = Runner(dsl=dsl)
        return self._fitness_metric(runner.run())

    def _fitness_metric(self, data):
        return data["average_execution"] * data["average_success"]

    def crossover(self, a: Genome, b: Genome) -> Tuple[Genome, Genome]:
        crossed_over = []
        for _a, _b in zip(a, b):
            indicator = random.random()
            if indicator > 0.5:
                crossed_over.append(_b)
            else:
                crossed_over.append(_a)

    def mutation(self, genome: Genome, func: MutationFunc = None) -> Genome:
        for i, entry in enumerate(genome):
            mutation_throw = random.random()
            if mutation_throw < self.mutation_probability:
                choices = [i for i in range(self.max_genome_value) if i != entry]
                genome[i] = random.choice(choices)

        return genome

    def selection_pair(self, population: Population) -> Tuple[Genome, Genome]:
        return None

    def create_new_generation(self, population: Population):
        fitness_list = list(map(lambda genome: self.fitness(genome), population))
        total_fitness = sum(fitness_list)
        probability_distribution = list(map(lambda fit: fit / total_fitness, fitness_list))
        new_population = []
        for _ in range(self.population_size):
            a, b = np.random.choice(population, 2, p=probability_distribution)
            child = self.mutation(self.crossover(a, b))
            new_population.append(child)

        return new_population

    def genome_to_string(self, genome: Genome) -> str:
        return "Genome(" + str(genome) + ")"

    def run_evolution(self) -> Tuple[Population, int]:
        population = self.generate_population()
        for _ in range(self.generation_limit):
            population = self.create_new_generation(population)
        return population
