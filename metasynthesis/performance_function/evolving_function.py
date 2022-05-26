import time
from operator import add
from typing import Tuple, List
from random import randint, random, choice
import numpy as np

from common.environment import RobotEnvironment
from common.program_synthesis.dsl import StandardDomainSpecificLanguage
from common.program_synthesis.objective import ObjectiveFun
from common.program_synthesis.runner import Runner
from metasynthesis.abstract_genetic import GeneticAlgorithm, MutationFunc, CrossoverFunc
from metasynthesis.performance_function.dom_dist_fun.pixel_dist_fun import PixelDistFun
from metasynthesis.performance_function.dom_dist_fun.robot_dist_fun import RobotDistFun
from metasynthesis.performance_function.dom_dist_fun.string_dist_fun import StringDistFun
from metasynthesis.performance_function.expr_tree import ExpressionTree, printN
from metasynthesis.performance_function.symbol import TermSym, OpSym
from search.brute.brute import Brute

Genome = ExpressionTree
Population = List[Genome]


class EvolvingFunction(GeneticAlgorithm):

    def __init__(self, fitness_limit: int, generation_limit: int, crossover_probability: float,
                 mutation_probability: float, generation_size: int,
                 domain: str, tournament_size: int = 2, elite_size: int = 6):
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

    def generate_genome(self, max_depth=4) -> Genome:
        return ExpressionTree.generate_random_expression(terms=self.terms, max_depth=max_depth)

    def generate_population(self, max_depth=4) -> Population:
        return [self.generate_genome(max_depth=max_depth) for _ in range(self.generation_size)]

    def fitness(self, genome: Genome) -> float:
        # data = Runner(StandardDomainSpecificLanguage(self.domain), Brute(0.1, genome.distance_fun)).run()
        # return data['average_success']
        return randint(0, 100)

    def crossover(self, a: Genome, b: Genome, func: CrossoverFunc = None) -> Tuple[Genome, Genome]:
        if random() < self.crossover_probability:
            return a.reproduce_with(other=b)
        else:
            return a, b

    def mutation(self, genome: Genome, func: MutationFunc = None) -> Genome:
        if random() < self.mutation_probability:
            return genome.mutate_tree()
        else:
            return genome

    def pop_fitness(self, pop: Population):
        # print('Calculating population fitness...')
        start_time = time.time()
        res = list(map(lambda x: self.fitness(x), pop))
        # print("--- %s seconds taken to calculate pop. fitness: ---" % (time.time() - start_time))
        return res
        # return list(map(lambda x: randint(0, 10), pop))

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

    def run_evolution(self, verbose=False) -> Tuple[List[float], List[float], Genome]:
        population = self.generate_population()
        avg_fit: List[float] = []
        best_fit: List[float] = []
        best_individual: Genome = None
        for cur_gen in range(self.generation_limit):
            # print('Starting generation no:', cur_gen)
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
            best_fit.append(cur_best)

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
        return avg_fit, best_fit, best_individual


def distance_manually_designed_function(env1: RobotEnvironment, env2: RobotEnvironment) -> float:
    def d(xy1: 'tuple[int, int]', xy2: 'tuple[int, int]'):
        return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

    # position robot and position ball
    pr = (env1.rx, env1.ry)
    pb = (env1.bx, env1.by)

    # position goal robot and position goal bal
    prstar = (env2.rx, env2.ry)
    pbstar = (env2.bx, env2.by)

    if d(pb, pbstar) == 0:
        if d(pr, prstar) == 0:
            if env1.holding == env2.holding:
                return 0
            else:
                return 1
        else:
            if env2.holding:
                return 1 + d(pr, prstar)
            else:
                return d(pr, prstar)
    else:
        if env1.holding:
            if not env2.holding:
                return 1 + d(pr, pbstar) + d(pbstar, prstar)
            else:
                return d(pr, pbstar)
        else:
            if not env2.holding:
                return 2 + d(pbstar, prstar) + d(pr, pb) + d(pb, pbstar)
            else:
                return 1 + d(pr, pb) + d(pb, pbstar)



def distance_default_expr_tree(env1: RobotEnvironment, env2: RobotEnvironment) -> float:
    def d(xy1: 'tuple[int, int]', xy2: 'tuple[int, int]'):
        return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

    # position robot and position ball
    pr = (env1.rx, env1.ry)
    pb = (env1.bx, env1.by)

    # position goal robot and position goal bal
    pgr = (env2.rx, env2.ry)
    pgb = (env2.bx, env2.by)

    partial_dist_funs = RobotDistFun.partial_dist_funs()
    terms = list(map(lambda x: TermSym(x), partial_dist_funs))

    if pr != pb and pb != pgb:
        root = ExpressionTree(OpSym(add), ExpressionTree(OpSym(add), ExpressionTree(terms[3], None, None),
                                                         ExpressionTree(terms[4], None, None)),
                              ExpressionTree(terms[2], None, None))
        return root.distance_fun(env1, env2) + 2
        # return d(pr, pb) + d(pb, pgb) + d(pgb, pgr) + 2
    elif pr == pb and pb != pgb:
        root = ExpressionTree(OpSym(add), ExpressionTree(terms[0], None, None), ExpressionTree(terms[2], None, None))
        return root.distance_fun(env1, env2) + 1
        # return d(pr, pgb) + d(pgb, pgr) + 1
    else:
        root = ExpressionTree(terms[1], None, None)
        return root.distance_fun(env1, env2)
        # return d(pr, pgr)


def rand_env() -> RobotEnvironment:
    size = 5
    rx = randint(0, size - 1)
    ry = randint(0, size - 1)
    bx = randint(0, size - 1)
    by = randint(0, size - 1)
    return RobotEnvironment(size=size, rx=rx, ry=ry, bx=bx, by=by, holding=False)


def test_default_vs_default_expr_tree():
    count = 0
    ct_1 = 0
    ct_2 = 0
    for i in range(10000):
        env1 = rand_env()
        env2 = rand_env()
        t1 = time.time()
        d1 = ObjectiveFun("robot").fun(env1, env2)
        ct_1 += time.time() - t1
        t2 = time.time()
        d2 = distance_default_expr_tree(env1, env2)
        ct_2 += time.time() - t2
        if d1 == d2:
            count += 1
    print(count == 10000)
    print(ct_1, ct_2)


if __name__ == '__main__':
    ga = EvolvingFunction(fitness_limit=0, generation_limit=20, crossover_probability=0.7,
                          mutation_probability=0.01, generation_size=40, domain='robot',
                          tournament_size=3, elite_size=2)
    avg_fit, best_fit, best_individual = ga.run_evolution()
    print(avg_fit)
    print(best_fit)
    print(best_individual)
    # T = time.time()
    # print(ga.fitness(ga.generate_genome(max_depth=2)))
    # print('TIME TAKEN:', time.time() - T)
    # test_default_vs_default_expr_tree()
