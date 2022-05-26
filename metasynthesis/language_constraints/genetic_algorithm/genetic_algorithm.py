import math
from copy import deepcopy
import numpy as np
import random
from typing import Tuple, List

from common.tokens.abstract_tokens import Token, EnvToken
from common.program_synthesis.runner import Runner
from common.program_synthesis.dsl import DomainSpecificLanguage, robot_tokens, pixel_tokens, string_tokens
from metasynthesis.abstract_genetic import GeneticAlgorithm, Population, Genome, MutationFunc
from metasynthesis.language_constraints.constraints.ConstraintFactory import ConstraintCombiner
from metasynthesis.language_constraints.constraints.Constraints import AbstractConstraint, InvalidSequenceException
from metasynthesis.language_constraints.genetic_algorithm.Constraint_Checker import Constraint_Walker


class ConstraintFunc:

    def __init__(self, genome, bool_tokens, trans_tokens, constraints):
        self.constraints = list(map(lambda c: deepcopy(c[0]).set_value(c[1]-1), filter(lambda c: c[1] == 0,  zip(constraints, genome))))
        self.constraints = ConstraintCombiner(self.constraints, trans_tokens).combine_all()
        self.bool_tokens = bool_tokens
        self.trans_tokens = trans_tokens


    def __call__(self, sequence: List[EnvToken]) -> bool:
        try:
            walker = Constraint_Walker(deepcopy(self.constraints))
            for token in sequence:
                walker(token)
            return True
        except InvalidSequenceException:
            return False


class ConstraintGeneticAlgorithm(GeneticAlgorithm):

    def __init__(self, fitness_limit: float,
                 generation_limit: int,
                 mutation_probability: float,
                 constraints: List[AbstractConstraint],
                 population_size: int = 20) -> object:
        super().__init__(fitness_limit, generation_limit, 0, mutation_probability)
        self.constraints = constraints
        self.max_genome_value = list(map(lambda c: c.get_values(), constraints))
        self.population_size = population_size
        self.fitness_memory = {}

    def generate_genome(self, length: int = -1) -> Genome:
        genome = map(lambda x: math.floor(random.random() * x), self.max_genome_value)
        return list(genome)

    def generate_population(self, size: int = -1) -> Population:
        return [self.generate_genome() for _ in range(self.population_size)]

    def _create_dsl(self, genome: Genome, domain):
        if domain == 'robot':
            module = robot_tokens
        elif domain == 'pixel':
            module = pixel_tokens
        else:
            module = string_tokens

        bt = module.BoolTokens
        tt = module.TransTokens
        func = ConstraintFunc(genome, bt, tt, self.constraints)

        return DomainSpecificLanguage(domain, bt, tt, True, func)

    def fitness(self, genome: Genome) -> float:
        if tuple(genome) in self.fitness_memory:
            return self.fitness_memory[tuple(genome)]
        dsl = self._create_dsl(genome, 'robot')
        runner = Runner(search_method=Brute(0.1), dsl=dsl)
        fitness = self._fitness_metric(runner.run())
        self.fitness_memory[tuple(genome)] = fitness
        print(fitness)
        return fitness

    def _fitness_metric(self, data):
        return (1/data["average_execution"]) * data["average_success"]**2

    def crossover(self, a: Genome, b: Genome) -> Tuple[Genome, Genome]:
        _1, _2 = [], []
        for _a, _b in zip(a, b):
            indicator = random.random()
            if indicator > 0.5:
                _1.append(_b)
                _2.append(_a)
            else:
                _1.append(_a)
                _2.append(_b)
        return _1, _2

    def mutation(self, genome: Genome, func: MutationFunc = None) -> Genome:
        for i, entry in enumerate(genome):
            mutation_throw = random.random()
            if mutation_throw < self.mutation_probability:
                choices = [j for j in range(self.max_genome_value[i]) if j != entry]
                genome[i] = random.choice(choices)

        return genome

    def create_new_generation(self, population: Population):
        fitness_list = list(map(lambda genome: self.fitness(genome), population))
        total_fitness = sum(fitness_list)
        probability_distribution = list(map(lambda fit: fit / total_fitness, fitness_list))
        new_population = []
        for _ in range(round(self.population_size/2)):
            a, b = np.random.choice(len(population), 2, p=probability_distribution)
            child1, child2 = self.crossover(population[a], population[b])
            child1 = self.mutation(child1)
            child2 = self.mutation(child2)
            new_population.append(child1)
            new_population.append(child2)

        return new_population

    def genome_to_string(self, genome: Genome) -> str:
        return "Genome(" + str(genome) + ")"

    def run_evolution(self):
        total_fitness = 0
        max_fitness = -1
        max_fitness_constraints = -1
        population = self.generate_population()
        for i in range(self.generation_limit):
            for pop in population:
                fitness = self.fitness(pop)
                print(self.genome_to_string(pop), self.fitness(pop))
                total_fitness += fitness
                if fitness > max_fitness:
                    max_fitness = fitness
                    max_fitness_constraints = sum([1 for gene in pop if gene > 0])

            population = self.create_new_generation(population)
            print(total_fitness/((i+1)*self.population_size), max_fitness, max_fitness_constraints)
        return sorted(map(lambda pop: (pop, self.fitness(pop)), population), 1)