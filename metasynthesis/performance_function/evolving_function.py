from typing import Tuple, List
from random import randint, random
import numpy as np

from metasynthesis.abstract_genetic import GeneticAlgorithm, MutationFunc, CrossoverFunc
from metasynthesis.performance_function.dom_dist_fun.pixel_dist_fun import PixelDistFun
from metasynthesis.performance_function.dom_dist_fun.robot_dist_fun import RobotDistFun
from metasynthesis.performance_function.dom_dist_fun.string_dist_fun import StringDistFun
from metasynthesis.performance_function.expr_tree import ExpressionTree, printN
from metasynthesis.performance_function.symbol import TermSym

Genome = ExpressionTree
Population = List[Genome]


class EvolvingFunction(GeneticAlgorithm):

    def __init__(self, fitness_limit: int, generation_limit: int, crossover_probability: float,
                 mutation_probability: float, generation_size: int,
                 domain_name: str, tournament_size: int = 2, elite_size: int = 6):
        assert (generation_size % 2 == 0), "population size (generation_size) should be even"
        assert (elite_size % 2 == 0) and elite_size < generation_size, "elite size (elite_size) should be even and (< gen_size)"
        super().__init__(fitness_limit=fitness_limit, generation_limit=generation_limit,
                         crossover_probability=crossover_probability,
                         mutation_probability=mutation_probability, generation_size=generation_size)
        self.domain = domain_name
        self.terms = None
        if domain_name == 'robot':
            partial_dist_funs = RobotDistFun.partial_dist_funs()
        elif domain_name == 'pixel':
            partial_dist_funs = PixelDistFun.partial_dist_funs()
        elif domain_name == 'string':
            partial_dist_funs = StringDistFun.partial_dist_funs()
        else:
            raise Exception('Domain should be one of: robot, string, pixel.')
        self.terms = list(map(lambda x: TermSym(x), partial_dist_funs))
        self.tournament_size = tournament_size
        self.elite_size = elite_size

    def generate_genome(self, max_depth=4) -> Genome:
        return ExpressionTree.generate_random_expression(terms=self.terms, max_depth=max_depth)

    def generate_population(self) -> Population:
        return [self.generate_genome() for _ in range(self.generation_size)]

    def fitness(self, genome: Genome) -> float:
        raise NotImplementedError()

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
        # return list(map(lambda x: self.fitness(x), pop))
        return list(map(lambda x: randint(0, 10), pop))

    def selection_pair(self, population: Population) -> Tuple[Genome, Genome]:
        fitness_values = self.pop_fitness(population)

        def tournament():
            n = len(fitness_values)
            best = None
            for _ in range(self.tournament_size):
                idx = randint(0, n - 1)
                if (best is None) or fitness_values[idx] > fitness_values[best]:
                    best = idx
            return best

        parents_idx = [tournament() for _ in range(2)]
        return population[parents_idx[0]], population[parents_idx[1]]

    def parent_selection(self, population: Population, num_pairs: int) -> List[Tuple[Genome, Genome]]:
        return [self.selection_pair(population) for _ in range(num_pairs)]

    def genome_to_string(self, genome: Genome) -> str:
        return str(genome)

    def run_evolution(self) -> Tuple[List[float], List[float], Genome]:
        population = self.generate_population()
        avg_fit: List[float] = []
        best_fit: List[float] = []
        best_individual: Genome = None
        for _ in range(self.generation_limit):
            fitness_values = np.array(self.pop_fitness(population))

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

            for parent_pair in self.parent_selection(population=population,
                                                 num_pairs=int((self.generation_size - self.elite_size) / 2)):
                p1, p2 = parent_pair[0], parent_pair[1]
                c1, c2 = self.crossover(p1, p2)
                c1_prime, c2_prime = self.mutation(genome=c1), self.mutation(genome=c2)
                offspring.append(c1_prime)
                offspring.append(c2_prime)

            population = offspring
        return avg_fit, best_fit, best_individual


if __name__ == '__main__':
    ga = EvolvingFunction(fitness_limit=0, generation_limit=100, crossover_probability=0.7,
                          mutation_probability=0.01, generation_size=30, domain_name='robot', tournament_size=3, elite_size=6)
    a = ga.generate_genome(max_depth=2)
    b = ga.generate_genome(max_depth=2)
    print(a)
    print(b)
    c1, c2 = ga.crossover(a=a, b=b)
    print(c1)
    print(c2)
    print(ga.mutation(genome=c2))
    selected1, selected2 = ga.selection_pair(ga.generate_population())
    print(selected1, '\n', selected2)

    par_sel = ga.parent_selection(ga.generate_population(), num_pairs=50)
    print(len(par_sel))
    for pair in par_sel:
        print(pair[0], '---------', pair[1])

    print(ga.run_evolution())
