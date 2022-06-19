import time
from operator import add
from typing import Tuple, List
from random import randint, random, choice
import numpy as np

from common.environment.robot_environment import RobotEnvironment
from common.program_synthesis.objective import ObjectiveFun
from metasynthesis.abstract_genetic import GeneticAlgorithm, MutationFunc, CrossoverFunc
from metasynthesis.performance_function.dom_dist_fun.pixel_dist_fun import PixelDistFun
from metasynthesis.performance_function.dom_dist_fun.robot_dist_fun import RobotDistFun
from metasynthesis.performance_function.dom_dist_fun.string_dist_fun import StringDistFun
from metasynthesis.performance_function.expr_tree import ExpressionTree
from metasynthesis.performance_function.symbol import TermSym, OpSym
from solver.runner.algorithms import dicts
from solver.runner.runner import Runner

Genome = ExpressionTree
Population = List[Genome]


class EvolvingFunction(GeneticAlgorithm):

    def __init__(self, fitness_limit: int, generation_limit: int, crossover_probability: float,
                 mutation_probability: float, generation_size: int,
                 domain: str, tournament_size: int = 2, elite_size: int = 6,
                 w1: float = 0.7, w2: float = 0.2, w3: float = 0.1,
                 d_max: int = 4, pr_op: float = 0.7, brute_time_out: float = 1.0):
        assert (generation_size % 2 == 0), "population size (generation_size) should be even"
        assert (elite_size % 2 == 0) and elite_size < generation_size, "elite size (elite_size) should be even and (< gen_size)"
        super().__init__(fitness_limit=fitness_limit, generation_limit=generation_limit,
                         crossover_probability=crossover_probability,
                         mutation_probability=mutation_probability, generation_size=generation_size)
        self.domain = domain
        self.terms = None
        if domain == 'robot':
            partial_dist_funs = RobotDistFun.partial_dist_funs()
        elif domain == 'pixel':
            partial_dist_funs = PixelDistFun.partial_dist_funs()
        elif domain == 'string':
            partial_dist_funs = StringDistFun.partial_dist_funs()
        else:
            raise Exception('Domain should be one of: robot, string, pixel.')
        self.terms = list(map(lambda x: TermSym(x), partial_dist_funs))
        self.tournament_size = tournament_size
        self.elite_size = elite_size
        self.fitness_dict = dict()
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.d_max = d_max
        self.pr_op = pr_op
        self.brute_time_out = brute_time_out

    def generate_genome(self) -> Genome:
        return ExpressionTree.generate_random_expression(terms=self.terms, max_depth=self.d_max, p=self.pr_op)

    def generate_population(self) -> Population:
        return [self.generate_genome() for _ in range(self.generation_size)]

    def fitness_manually_designed_fun(self):

        if self.domain == 'robot':
            percentage_of_examples_solved, average_norm_run_time = Runner(lib=dicts(), algo='Brute',
                                                                          setting='RO', test_cases='eval',
                                                                          time_limit_sec=self.brute_time_out,
                                                                          debug=False, store=False).run()
            return self.w1 * percentage_of_examples_solved + self.w3 * (1 - average_norm_run_time)
        elif self.domain == 'string':
            percentage_of_solved_tasks, mean_perc_unsolved_ex_unsolved_tasks, average_norm_run_time = Runner(dicts(),
                                                                                                             'Brute',
                                                                                                             setting='SO',
                                                                                                             test_cases='eval',
                                                                                                             time_limit_sec=self.brute_time_out,
                                                                                                             debug=False,
                                                                                                             store=False).run(
                                                                                                             alternative_fitness=True)
            fit_val = self.w1 * percentage_of_solved_tasks + \
                      self.w2 * (1 - mean_perc_unsolved_ex_unsolved_tasks) + \
                      self.w3 * (1 - average_norm_run_time)
            return fit_val


    def fitness(self, genome: Genome, test_cases='param') -> float:
        if genome in self.fitness_dict:
            return self.fitness_dict[genome]
        else:
            setting = None
            if self.domain == 'robot':
                setting = 'RO'
            elif self.domain == 'string':
                setting = 'SO'

            if self.domain == 'robot':
                percentage_of_examples_solved, average_norm_run_time = Runner(dicts(), 'Brute', setting=setting,
                                                                              test_cases=test_cases,
                                                                              time_limit_sec=self.brute_time_out,
                                                                              debug=False, store=False,
                                                                              dist_fun=genome.distance_fun).run(
                                                                              alternative_fitness=False)
                fit_val = self.w1 * percentage_of_examples_solved + self.w3 * (1 - average_norm_run_time)
                self.fitness_dict[genome] = fit_val
                return fit_val
            elif self.domain == 'string':
                percentage_of_solved_tasks, mean_perc_unsolved_ex_unsolved_tasks, average_norm_run_time = Runner(dicts(), 'Brute', setting=setting,
                                                                              test_cases=test_cases,
                                                                              time_limit_sec=self.brute_time_out,
                                                                              debug=False, store=False,
                                                                              dist_fun=genome.distance_fun).run(
                                                                              alternative_fitness=True)
                fit_val = self.w1 * percentage_of_solved_tasks + \
                          self.w2 * (1 - mean_perc_unsolved_ex_unsolved_tasks) + \
                          self.w3 * (1 - average_norm_run_time)
                self.fitness_dict[genome] = fit_val
                return fit_val

    def crossover(self, a: Genome, b: Genome, func: CrossoverFunc = None) -> Tuple[Genome, Genome]:
        if random() < self.crossover_probability:
            return a.reproduce_with(other=b)
        else:
            return a, b

    def mutation(self, genome: Genome, func: MutationFunc = None) -> Genome:
        if random() < self.mutation_probability:
            return genome.mutate_tree(domain=self.domain)
        else:
            return genome

    def pop_fitness(self, pop: Population):
        res = list(map(lambda x: self.fitness(x), pop))
        return res

    def selection_pair(self, population: Population, fitness_values: List[float]) -> Tuple[Genome, Genome]:

        def tournament(idx_to_exclude: int):
            if idx_to_exclude == -1:  # first parent
                n = len(fitness_values)
                best = None
                for _ in range(self.tournament_size):
                    idx = randint(0, n - 1)
                    if (best is None) or fitness_values[idx] > fitness_values[best]:
                        best = idx
                return best
            else:
                # the first parent should not be considered again
                n = len(fitness_values)
                best = None
                updated = list(range(n))
                updated.remove(idx_to_exclude)
                for _ in range(self.tournament_size):
                    idx = choice(updated)
                    if (best is None) or fitness_values[idx] > fitness_values[best]:
                        best = idx
                return best

        p1_idx = tournament(idx_to_exclude=-1)
        p2_idx = tournament(idx_to_exclude=p1_idx)

        # parents_idx = [tournament() for _ in range(2)]
        return population[p1_idx], population[p2_idx]

    def parent_selection(self, population: Population, fitness_values, num_pairs: int) -> List[Tuple[Genome, Genome]]:
        return [self.selection_pair(population, fitness_values) for _ in range(num_pairs)]

    def genome_to_string(self, genome: Genome) -> str:
        return str(genome)

    def run_evolution(self, verbose=False) -> Tuple[List[float], List[float], Genome, float, float]:
        start_time = time.time()
        population = self.generate_population()
        avg_fit: List[float] = []
        best_fit: List[float] = []
        best_individual: Genome = None
        for cur_gen in range(self.generation_limit):
            print('Starting generation no:', cur_gen, 'at second:', time.time() - start_time)
            fitness_values = np.array(self.pop_fitness(population))
            heights = np.array([x.height() for x in population])
            if verbose:
                print('Average height of generation:', cur_gen, ':', np.mean(heights))
                print('MIN height', np.min(heights))
                print('MAX height', np.max(heights))
                print('---------------')

            ind_ranked = np.argsort(-fitness_values)
            top_k_ind = ind_ranked[:self.elite_size]
            best_individual = population[top_k_ind[0]]
            avg_fit.append(np.mean(fitness_values))
            cur_best = np.max(fitness_values)

            print('Max fitness in generation', cur_gen, ':', cur_best)
            best_fit.append(cur_best)

            if cur_gen == self.generation_limit - 1:
                break

            offspring = []
            i = 0

            while i < self.elite_size:
                offspring.append(population[top_k_ind[i]])
                i += 1

            for parent_pair in self.parent_selection(population=population, fitness_values=fitness_values,
                                                     num_pairs=int((self.generation_size - self.elite_size) / 2)):
                p1, p2 = parent_pair[0], parent_pair[1]
                c1, c2 = self.crossover(p1, p2)
                c1_prime, c2_prime = self.mutation(genome=c1), self.mutation(genome=c2)
                offspring.append(c1_prime)
                offspring.append(c2_prime)

            population = offspring
        del self.fitness_dict[best_individual]
        return avg_fit, best_fit, best_individual, \
               self.fitness(genome=best_individual, test_cases='eval'), self.fitness_manually_designed_fun()
