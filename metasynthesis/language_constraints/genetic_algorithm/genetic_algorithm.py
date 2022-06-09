import math
from copy import deepcopy
import numpy as np
import random
from typing import Tuple, List

from common.tokens.abstract_tokens import EnvToken
from metasynthesis.language_constraints.genetic_algorithm.geneticDataManager import DataManager
from solver.runner.algorithms import dicts
from solver.runner.runner import Runner
from common.program_synthesis.dsl import DomainSpecificLanguage, StandardDomainSpecificLanguage
from metasynthesis.abstract_genetic import GeneticAlgorithm, Population, Genome, MutationFunc
from metasynthesis.language_constraints.constraints.ConstraintFactory import ConstraintCombiner
from metasynthesis.language_constraints.constraints.Constraints import AbstractConstraint, InvalidSequenceException
from metasynthesis.language_constraints.genetic_algorithm.Constraint_Checker import ConstraintBuffer
from solver.invent.static_invent import StaticInvent


class ConstraintFunc:

    def __init__(self, genome, dsl: DomainSpecificLanguage, constraints):
        self.constraints = list(map(lambda c: deepcopy(c[0]).set_value(c[1]-1), filter(lambda c: c[1] != 0,  zip(constraints, genome))))
        self.constraints = ConstraintCombiner(self.constraints, dsl.get_trans_tokens()).combine_all()
        si = StaticInvent()
        si.setup(dsl)
        tokens = si.perms + si.ifs + si.loops
        self.checker = ConstraintBuffer(self.constraints, tokens)

    def __call__(self, sequence: List[EnvToken]) -> bool:
        try:
            self.checker.check_token(sequence)
            return True
        except InvalidSequenceException:
            return False


class ConstraintGeneticAlgorithm(GeneticAlgorithm):

    def __init__(self, fitness_limit: float,
                 generation_limit: int,
                 mutation_probability: float,
                 constraints: List[AbstractConstraint],
                 algorithm_settings: dict,
                 population_size: int = 20):
        super().__init__(fitness_limit, generation_limit, 0, mutation_probability, population_size)
        self.constraints = constraints
        self.max_genome_value = list(map(lambda c: c.get_values(), constraints))
        self.population_size = population_size
        self.fitness_memory = {}
        self.settings = algorithm_settings
        self.best_chromosome = ()
        runner = Runner(dicts(),
                        self.settings['algorithm'],
                        self.settings['setting'],
                        self.settings['test_cases'],
                        self.settings['time_limit'],
                        self.settings['debug'],
                        self.settings['store'],
                        dsl=StandardDomainSpecificLanguage(self.settings['domain']))
        runner.run()
        self.fitness_bias = 0
        self.fitness_bias = self._fitness_metric(runner.file_manager.written_data)
        self.logger = DataManager(self.settings['domain'], self.settings['algorithm'], self.settings['setting'])

    def generate_genome(self, length: int = -1) -> Genome:
        genome = [random.choice(range(max_value+1)) for max_value in self.max_genome_value]
        return list(genome)

    def generate_population(self, size: int = -1) -> Population:
        return [self.generate_genome() for _ in range(self.population_size)]

    def _create_dsl(self, genome: Genome, domain):
        main_dsl = StandardDomainSpecificLanguage(domain)
        func = ConstraintFunc(genome, main_dsl, self.constraints)
        return DomainSpecificLanguage(domain, main_dsl.get_bool_tokens(), main_dsl.get_trans_tokens(), True, func)

    def fitness(self, genome: Genome, bias=True) -> float:
        if tuple(genome) in self.fitness_memory:
            return self.fitness_memory[tuple(genome)]
        dsl = self._create_dsl(genome, self.settings['domain'])
        runner = Runner(dicts(),
                        self.settings['algorithm'],
                        self.settings['setting'],
                        self.settings['test_cases'],
                        self.settings['time_limit'],
                        self.settings['debug'],
                        self.settings['store'],
                        dsl=dsl)
        runner.run()
        fitness = self._fitness_metric(runner.file_manager.written_data, bias)
        self.logger.write_chromosome_data(genome, runner.file_manager.written_data)
        self.fitness_memory[tuple(genome)] = fitness
        return fitness

    def _fitness_metric(self, data, bias=True):
        programs_considered = 0
        cost = 0
        execution_time = 0
        max_cost = 0
        for d in data[0]:
            programs_considered += d['no._explored_programs']
            cost += d['train_cost']+d['test_cost']
            max_cost = max(d['train_cost']+d['test_cost'], max_cost)
            execution_time += d['execution_time']
        cost /= len(data[0])
        execution_time /= len(data[0])
        if bias:
            fit = (1/(cost*max_cost*execution_time)**(1/2)) * 100 - self.fitness_bias
        else:
            fit = (1 / (cost * max_cost * execution_time) ** (1 / 2)) * 100
        return max(fit, 0)

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
                choices = [j for j in range(self.max_genome_value[i]+1) if j != entry]
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
        population = self.generate_population()
        logger_data = {}
        bias = True
        for i in range(self.generation_limit):
            logger_data[i] = []
            for pop in population:
                fitness = self.fitness(pop, bias)
                print(self.genome_to_string(pop), self.fitness(pop))
                total_fitness += fitness
                if fitness > max_fitness:
                    max_fitness = fitness
                    self.best_chromosome = (pop, max_fitness)
                logger_data[i].append(("pop_report", pop, fitness))

            if total_fitness == 0 and bias:
                bias = False
                self.fitness_bias = {}
                logger_data[i] = []
                logger_data[i].append(("bias_disabled",))
                for pop in population:
                    fitness = self.fitness(pop, bias)
                    print(self.genome_to_string(pop), self.fitness(pop))
                    total_fitness += fitness
                    if fitness > max_fitness:
                        max_fitness = fitness
                        self.best_chromosome = (pop, max_fitness)

            logger_data[i].append(("general", self.best_chromosome))
            self.logger.write_genetic_stats(logger_data)

            population = self.create_new_generation(population)
        return self._create_dsl(self.best_chromosome[0], self.settings['domain'])
