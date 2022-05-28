import copy
import random
from random import randrange
from typing import List, Callable, Tuple, Iterable

from common.program_synthesis.dsl import DomainSpecificLanguage, StandardDomainSpecificLanguage
from solver.runner.runner import Runner
from solver.runner.algorithms import dicts


from metasynthesis.abstract_genetic import GeneticAlgorithm

Genome = DomainSpecificLanguage
Population = List[Genome]

genome_fitness_values = {}


class EvolvingLanguage(GeneticAlgorithm):
    """A genetic algorithm to synthesise a programing language for program synthesis"""

    def __init__(self, fitness_limit: int, generation_limit: int, crossover_probability: float,
                 elite_genomes: int, mutation_probability: float, generation_size: int,
                 dsl: DomainSpecificLanguage, search_setting: str, max_search_time: float,
                 search_mode: str):
        super().__init__(fitness_limit, generation_limit, crossover_probability, mutation_probability, generation_size)
        self.domain = dsl.domain_name
        self.dsl = dsl
        self.bool_tokens = dsl.get_bool_tokens()
        self.elite_genomes = elite_genomes
        self.trans_tokens = dsl.get_trans_tokens()
        self.search_setting = search_setting
        self.max_search_time = max_search_time
        self.search_mode = search_mode

    def generate_genome(self, length: int) -> Genome:
        """This method creates a new genome of the specified length"""

        max_dsl_length = len(self.bool_tokens + self.trans_tokens)
        bool_length = round((length / max_dsl_length) * len(self.bool_tokens))
        trans_length = round((length / max_dsl_length) * len(self.trans_tokens))

        bool_chromosomes = random.sample(self.bool_tokens, bool_length)
        trans_chromosomes = random.sample(self.trans_tokens, trans_length)

        dsl_genome = DomainSpecificLanguage(self.domain, bool_chromosomes, trans_chromosomes)

        return sort_genome(dsl_genome)

    def generate_population(self) -> Population:
        """This method creates a population of new genomes"""

        max_dsl_length = len(self.bool_tokens + self.trans_tokens)

        population = []

        for i in range(0, self.generation_size):
            genome_length = randrange(1, max_dsl_length + 1)
            population.append(self.generate_genome(genome_length))

        return population

    def fitness(self, genome: Genome) -> float:
        """This method calculates the fitness of the specified genome"""

        # need a combination of
        # 1) length of genome
        # 2) whether program that solves tasks was created
        # 3) number of iterations of searching

        if str(genome) in genome_fitness_values.keys():
            # print("ALREADY IN", str(genome))
            return genome_fitness_values[str(genome)]

        # genome = StandardDomainSpecificLanguage("robot")
        runner = Runner(lib=dicts(0),
                        algo="AS",
                        setting=self.search_setting,
                        test_cases=self.search_mode,
                        time_limit_sec=self.max_search_time,
                        debug=False,
                        store=False,
                        suffix="",
                        dsl=genome)
        runner.run()

        search_results = runner.search_results

        total_cases = 0
        cumulative_ratios_correct = 0
        cumulative_search_time_correct = 0

        for key, value in search_results.items():
            total_cases += 1
            current_search_result = value
            search_time = current_search_result["search_time"]
            test_total = current_search_result["test_total"]

            current_ratio_correct = 0
            # If we use string domain we should use tests, instead of training examples
            if self.domain == "string":
                test_correct = current_search_result["test_correct"]
                current_ratio_correct += test_correct / test_total
            else:
                train_correct = current_search_result["train_correct"]
                current_ratio_correct += train_correct / test_total

            cumulative_ratios_correct += current_ratio_correct
            if current_ratio_correct > 0:
                cumulative_search_time_correct += search_time
            else:
                cumulative_search_time_correct += self.max_search_time

        mean_ratio_correct = cumulative_ratios_correct / total_cases
        mean_search_time_correct = cumulative_search_time_correct / total_cases

        if mean_search_time_correct == 0:
            fitness_value = 0
        else:
            fitness_value = mean_ratio_correct * (1 / mean_search_time_correct)

        print("correct:", mean_ratio_correct, "search:", mean_search_time_correct, "fitness:", fitness_value)

        # avg_exec_time = results["average_execution"]
        # inverse_avg_exec_time = 1 / (avg_exec_time + 0.000000001)
        #
        # success_percentage = results["average_success"]
        # success_percentage_scaled = success_percentage / 100
        #
        # # print(inverse_dsl_length, inverse_avg_exec_time, success_percentage_scaled)
        #
        # fitness_value = inverse_dsl_length * inverse_avg_exec_time * success_percentage_scaled
        #


        genome_fitness_values[str(genome)] = fitness_value

        return fitness_value

    def crossover(self, a: Genome, b: Genome) -> Tuple[Genome, Genome]:
        """This method applies the given crossover function with certain probability"""

        rand_float = random.random()
        crossed_a = None
        crossed_b = None

        if 0 <= rand_float <= 1:
            crossed_a, crossed_b = crossover_exchange_trans_bool(a, b)

        return crossed_a, crossed_b

    def mutation(self, genome: Genome) -> Genome:
        """This method applies mutation to a genome with certain probability"""

        rand_float = random.random()
        mutated_genome = None

        if 0 <= rand_float <= 0.5:
            mutated_genome = mutate_add_token(genome, self.dsl)
        elif 0.5 < rand_float <= 1:
            mutated_genome = mutate_remove_token(genome)

        return mutated_genome

    def sort_population(self, population: Population) -> Population:
        """This method sorts the population based on the given fitness function"""

        return sorted(population, key=lambda genome: self.fitness(genome), reverse=True)

    def genome_to_string(self, genome: Genome) -> str:
        """This method converts a given genome to a string"""

        raise NotImplementedError()

    def run_evolution(self):
        """This method runs the evolution process"""

        full_dsl = sort_genome(StandardDomainSpecificLanguage(self.domain))
        print("FULL DSL FITNESS", self.fitness(full_dsl))

        iteration_count = 0
        population = self.generate_population()

        while iteration_count < self.generation_limit:

            # STATS
            print("GENERATION:", iteration_count + 1, "/", self.generation_limit)
            for genome in population:
                # self.fitness(genome)
                print(round(self.fitness(genome), 5), str(genome))

            new_population = copy.deepcopy(population)

            generation_cum_fitness = 0
            for genome in population:
                generation_cum_fitness += self.fitness(genome)
            print("AVERAGE GENERATION FITNESS", generation_cum_fitness / self.generation_size)
            # print("POP LENGTH", len(population))

            best_percentage = select_best_percentage(population=new_population, percentage=50,
                                                     size=self.generation_size)

            # ELITISM
            for elite_index in range(0, self.elite_genomes):
                new_population[elite_index] = best_percentage[elite_index]

            random_crossover_probability = random.random()
            # CROSSOVER
            if random_crossover_probability < self.crossover_probability:
                # TODO: now we take crossover probability as if we apply it on the whole generation, want to apply per genome (pair)

                crossed_result = []
                genome_count = 0
                while genome_count < len(best_percentage) - 1:
                    a = best_percentage[genome_count]
                    b = best_percentage[genome_count + 1]
                    crossed_a, crossed_b = self.crossover(a, b)
                    genome_count += 2
                    crossed_result = crossed_result + [a, b, crossed_a, crossed_b]
                new_population[self.elite_genomes:self.generation_size] = crossed_result[
                                                                          0:self.generation_size - self.elite_genomes]

            # MUTATION
            for genome_index in range(self.elite_genomes, self.generation_size):
                random_mutation_probability = random.random()
                if random_mutation_probability < self.mutation_probability:
                    mutated_genome = self.mutation(copy.deepcopy(new_population[genome_index]))
                    new_population[genome_index] = mutated_genome
            iteration_count += 1

            population = new_population

        best_genome = population[0]

        print("ORIGINAL DSL")
        self.final_evaluation(full_dsl)
        print("EVOLVED DSL")
        self.final_evaluation(best_genome)

        return best_genome

    def final_evaluation(self, genome: Genome):
        runner = Runner(dicts(0),
                        algo="Brute",
                        setting=self.search_setting,
                        test_cases=self.search_mode,
                        time_limit_sec=self.max_search_time,
                        debug=False,
                        store=False,
                        suffix="",
                        dsl=genome)
        perc_solved = runner.run()

        print(perc_solved)

        # success_percentage = results["completely_successful_percentage"]
        # average_execution_time = results["average_execution"]
        #
        # search_results = results["programs"]
        # cum_program_length = 0
        # total_search_time = 0
        # for result in search_results:
        #     total_search_time += result.dictionary["execution_time"]
        #     cum_program_length += result.dictionary["program_length"]
        #
        # avg_program_length = cum_program_length / len(search_results)
        # print("Total search time", total_search_time)
        # print("Success percentage:", success_percentage)
        # print("Average program runtime:", average_execution_time)
        # print("Average program length", avg_program_length)


def sort_genome(genome: Genome) -> Genome:
    sorted_bool = sorted(genome.get_bool_tokens(), key=str, reverse=False)
    sorted_trans = sorted(genome.get_trans_tokens(), key=str, reverse=False)

    return DomainSpecificLanguage(genome.domain_name, sorted_bool, sorted_trans)


def genome_length(genome: Genome) -> int:
    return len(genome.get_bool_tokens() + genome.get_trans_tokens())


def get_fitness(genome: Genome):
    return genome_fitness_values[str(genome)]


def select_best_percentage(population: Population, percentage: float, size: int) -> Population:
    sorted_population = sorted(population, key=lambda x: get_fitness(x), reverse=True)

    max_to_get = round((percentage / 100) * size)

    for genome_index in range(0, len(sorted_population)):
        sorted_population[genome_index] = sort_genome(sorted_population[genome_index])

    return sorted_population[:max_to_get]


# CROSSOVER FUNCTIONS
def crossover_exchange_trans_bool(a: Genome, b: Genome) -> Tuple[Genome, Genome]:
    """Simply crosses the trans and bool tokens of two genomes"""

    crossed_a = DomainSpecificLanguage(a.domain_name, a.get_bool_tokens(), b.get_trans_tokens())
    crossed_b = DomainSpecificLanguage(a.domain_name, b.get_bool_tokens(), a.get_trans_tokens())

    return crossed_a, crossed_b


# MUTATION FUNCTIONS
def mutate_add_token(genome: Genome, dsl: DomainSpecificLanguage) -> Genome:
    bool_tokens_genome = genome.get_bool_tokens()
    trans_tokens_genome = genome.get_trans_tokens()

    all_tokens_dsl = dsl.get_bool_tokens() + dsl.get_trans_tokens()
    all_tokens_genome = bool_tokens_genome + trans_tokens_genome

    if len(all_tokens_dsl) == len(all_tokens_genome):
        return DomainSpecificLanguage(genome.domain_name, bool_tokens_genome, trans_tokens_genome)

    optional_tokens = []
    for token in all_tokens_dsl:
        if token not in all_tokens_genome:
            optional_tokens.append(token)

    random_index = randrange(0, len(optional_tokens))
    randomly_selected_token = optional_tokens[random_index]

    if randomly_selected_token in dsl.get_bool_tokens():
        # Picked a bool token
        bool_tokens_genome.append(randomly_selected_token)
    else:
        # Picked a trans token
        trans_tokens_genome.append(randomly_selected_token)

    result = sort_genome(DomainSpecificLanguage(genome.domain_name, bool_tokens_genome, trans_tokens_genome))
    return result


def mutate_remove_token(genome: Genome) -> Genome:
    bool_tokens_genome = genome.get_bool_tokens()
    trans_tokens_genome = genome.get_trans_tokens()

    all_tokens_genome = bool_tokens_genome + trans_tokens_genome

    if len(all_tokens_genome) == 0:
        return DomainSpecificLanguage(genome.domain_name, bool_tokens_genome, trans_tokens_genome)

    random_index = randrange(0, len(all_tokens_genome))
    randomly_selected_token = all_tokens_genome[random_index]

    if randomly_selected_token in bool_tokens_genome and len(bool_tokens_genome) > 0:
        # Picked a bool token and check if minimally one token
        bool_tokens_genome.remove(randomly_selected_token)
    elif randomly_selected_token in trans_tokens_genome and len(trans_tokens_genome) > 0:
        # Picked a trans token and check if minimally one token
        trans_tokens_genome.remove(randomly_selected_token)
    result = sort_genome(DomainSpecificLanguage(genome.domain_name, bool_tokens_genome, trans_tokens_genome))
    return result

