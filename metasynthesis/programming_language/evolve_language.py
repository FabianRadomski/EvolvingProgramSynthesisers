import copy
import random

import time
from random import randrange
from typing import List, Tuple

from common.program_synthesis.dsl import DomainSpecificLanguage, StandardDomainSpecificLanguage
from common.tokens.abstract_tokens import Token
from solver.runner.runner import Runner
from solver.runner.algorithms import dicts

from metasynthesis.abstract_genetic import GeneticAlgorithm

Genome = DomainSpecificLanguage
Population = List[Genome]

genome_fitness_values = {}
successful_tokens_counts = {}
successful_tokens_weights = {}
successful_tokens_objects = {}


class EvolvingLanguage(GeneticAlgorithm):
    """A genetic algorithm to synthesise a programing language for program synthesis"""

    def __init__(self, fitness_limit: int = 1, generation_limit: int = 20, crossover_probability: float = 0.8,
                 elite_genomes: int = 2, mutation_probability: float = 0.2, generation_size: int = 20,
                 dsl: DomainSpecificLanguage = StandardDomainSpecificLanguage("string"), search_setting: str = "SO",
                 max_search_time: float = 1, search_mode: str = "debug", search_algo: str = "Brute",
                 mutation_weights: List = (0.45, 0.05, 0.5), crossover_weights: List = (0, 0, 1),
                 start_population: Population = None, print_stats: bool = True):
        super().__init__(fitness_limit, generation_limit, crossover_probability, mutation_probability, generation_size)
        self.domain = dsl.domain_name
        self.dsl = dsl
        self.bool_tokens = dsl.get_bool_tokens()
        self.elite_genomes = elite_genomes
        self.trans_tokens = dsl.get_trans_tokens()
        self.search_setting = search_setting
        self.max_search_time = max_search_time
        self.search_mode = search_mode
        self.search_algo = search_algo
        self.mutation_weights = mutation_weights
        self.crossover_weights = crossover_weights
        self.start_population = start_population
        self.print_stats = print_stats

        self.full_dsl_correct_ratio = 0.0

    def generate_genome(self, length: int) -> Genome:
        """This method creates a new genome of the specified length"""

        all_tokens = self.bool_tokens + self.trans_tokens

        selected_tokens = random.sample(all_tokens, length)

        bool_tokens = [x for x in selected_tokens if x not in self.trans_tokens]
        trans_tokens = [x for x in selected_tokens if x not in self.bool_tokens]

        dsl_genome = DomainSpecificLanguage(self.domain, bool_tokens, trans_tokens)

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

        if str(genome) in genome_fitness_values.keys():
            return genome_fitness_values[str(genome)]["fitness"]

        runner = Runner(lib=dicts(0),
                        algo=self.search_algo,
                        setting=self.search_setting,
                        test_cases=self.search_mode,
                        time_limit_sec=self.max_search_time,
                        debug=False,
                        store=False,
                        suffix="",
                        dsl=genome)
        runner.run()

        mean_ratio_correct_original, mean_search_time, best_programs = \
            process_search_results(runner.search_results, self.domain)

        mean_ratio_correct = mean_ratio_correct_original

        # To prevent high fitness for DSL's that solve only a few tasks with extremely low search times
        if mean_ratio_correct < self.full_dsl_correct_ratio / 2:
            mean_ratio_correct = 0.0

        # To prevent division by zero
        if mean_search_time == 0:
            fitness_value = 0
        else:
            fitness_value = mean_ratio_correct * (1 / mean_search_time)

        extract_special_tokens(best_programs, self.dsl)

        genome_fitness_values[str(genome)] = {"genome": genome, "correct": mean_ratio_correct,
                                              "search": mean_search_time, "fitness": fitness_value}

        return fitness_value

    def crossover(self, a: Genome, b: Genome) -> Tuple[Genome, Genome]:
        """This method applies the given crossover function with certain probability"""

        rand_float = random.random()
        crossed_a = None
        crossed_b = None

        if rand_float <= self.crossover_weights[0]:
            crossed_a, crossed_b = crossover_exchange_trans_bool(a, b)
        elif rand_float <= self.crossover_weights[0] + self.crossover_weights[1]:
            crossed_a, crossed_b = crossover_exchange_halve_category(a, b)
        elif rand_float <= self.crossover_weights[0] + self.crossover_weights[1] + self.crossover_weights[2]:
            crossed_a, crossed_b = crossover_exchange_halve_random(a, b)

        return crossed_a, crossed_b

    def mutation(self, genome: Genome) -> Genome:
        """This method applies mutation to a genome with certain probability"""

        rand_float = random.random()
        mutated_genome = genome

        if rand_float <= self.mutation_weights[0]:
            mutated_genome = mutate_add_special_token(genome)
        elif rand_float <= self.mutation_weights[0] + self.mutation_weights[1]:
            mutated_genome = mutate_add_token(genome, self.dsl)
        elif rand_float <= self.mutation_weights[0] + self.mutation_weights[1] + self.mutation_weights[2]:
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

        genome_fitness_values.clear()
        self.fitness(sort_genome(StandardDomainSpecificLanguage(self.domain)))
        self.full_dsl_correct_ratio = \
            genome_fitness_values[str(sort_genome(StandardDomainSpecificLanguage(self.domain)))]["correct"]

        t1_start_all = time.perf_counter()

        full_dsl = sort_genome(StandardDomainSpecificLanguage(self.domain))
        self.fitness(full_dsl)
        if self.print_stats:
            print("FULL DSL:")
            print_chromosome_stats(full_dsl)

        generation_count = 0
        if self.start_population is None:
            population = self.generate_population()
        else:
            population = self.start_population
        generation_statistics = {}

        while generation_count < self.generation_limit:
            t2_start_generation = time.perf_counter()
            if self.print_stats:
                print("GENERATION:", generation_count + 1, "/", self.generation_limit)

            # EVALUATING CHROMOSOMES
            for chromosome in population:
                self.fitness(chromosome)
                if self.print_stats:
                    print_chromosome_stats(chromosome)

            # GENERATION SETUP
            successful_tokens_weights.clear()
            new_population = copy.deepcopy(population)
            sorted_population = sorted(new_population, key=lambda x: get_fitness(x), reverse=True)

            # ELITISM
            elite_chromosomes = []
            for elite_index in range(0, self.elite_genomes):
                elite_chromosomes.append(sorted_population[elite_index])

            # CROSSOVER
            crossed_chromosomes = self.apply_crossover(sorted_population)
            elite_and_crossed = elite_chromosomes + crossed_chromosomes

            # MUTATION
            normalize_special_token_weights()
            elite_crossed_mutated = elite_and_crossed[:self.elite_genomes] \
                                    + self.apply_mutation(elite_and_crossed[self.elite_genomes:])

            # GENERATION STATISTICS
            generation_statistics[str(generation_count)] = \
                self.calculate_generation_statistics(population, t2_start_generation)

            # PRINTING STATISTICS
            generation_average_fitness = generation_statistics[str(generation_count)]["Average fitness"]
            generation_time_taken = generation_statistics[str(generation_count)]["Time taken"]
            if self.print_stats:
                print("AVG FITNESS:", round(generation_average_fitness, 4),
                      "TIME TAKEN:", generation_time_taken)

            population = elite_crossed_mutated
            generation_count += 1

        best_genome = population[0]

        if self.print_stats:
            print("ORIGINAL DSL")
        final_original_results = self.final_evaluation(full_dsl)
        if self.print_stats:
            print("EVOLVED DSL")
        final_evolved_results = self.final_evaluation(best_genome)
        t1_stop_all = time.perf_counter()

        total_time_taken = t1_stop_all - t1_start_all
        num_explored_languages = len(genome_fitness_values.keys())

        if self.print_stats:
            print("Elapsed time during the whole program in seconds:", total_time_taken)
            print("Explored languages:", num_explored_languages)

        return {"Total time taken": total_time_taken,
                "Explored languages": num_explored_languages,
                "Final original results": final_original_results,
                "Final evolved results": final_evolved_results,
                "Generation statistics": generation_statistics}

    def apply_crossover(self, sorted_population: Population):
        crossed_chromosomes = []
        two_crossover_candidates = []
        for chromosome in sorted_population:
            random_crossover_chance = random.random()
            if random_crossover_chance < self.crossover_probability:
                two_crossover_candidates.append(chromosome)
                if len(two_crossover_candidates) == 2:
                    a, b = self.crossover(two_crossover_candidates[0], two_crossover_candidates[1])
                    crossed_chromosomes += [a, b]
                    sorted_population.remove(two_crossover_candidates[0])
                    sorted_population.remove(two_crossover_candidates[1])
                    two_crossover_candidates = []

        num_still_to_add = self.generation_size - self.elite_genomes - len(crossed_chromosomes)
        crossed_chromosomes += sorted_population[:num_still_to_add]
        return crossed_chromosomes

    def apply_mutation(self, elite_and_crossed: Population):
        for chromosome_index in range(self.elite_genomes, self.generation_size):
            random_mutation_probability = random.random()
            if random_mutation_probability < self.mutation_probability:
                mutated_genome = self.mutation(elite_and_crossed[chromosome_index])
                elite_and_crossed[chromosome_index] = mutated_genome
        return elite_and_crossed

    def calculate_generation_statistics(self, population: Population, t2_start_generation: float):
        generation_cumulative_fitness = 0
        for chromosome in population:
            generation_cumulative_fitness += self.fitness(chromosome)

        t2_stop_generation = time.perf_counter()
        generation_time_taken = t2_stop_generation - t2_start_generation
        generation_average_fitness = generation_cumulative_fitness / self.generation_size
        generation_best_fitness = self.fitness(population[0])

        generation_best_stats = get_chromosome_stats(population[0])
        return {"Time taken": generation_time_taken,
                "Best fitness": generation_best_fitness,
                "Average fitness": generation_average_fitness,
                "Generation best stats": generation_best_stats}

    def final_evaluation(self, genome: Genome):
        runner = Runner(dicts(0),
                        algo=self.search_algo,
                        setting=self.search_setting,
                        test_cases="param_test",
                        time_limit_sec=self.max_search_time,
                        debug=False,
                        store=False,
                        suffix="",
                        dsl=genome)
        runner.run()

        mean_ratio_correct, mean_search_time_correct, best_programs = \
            process_search_results(runner.search_results, self.domain)

        avg_program_length = sum(map(lambda program: program.number_of_tokens(), best_programs)) / len(best_programs)

        if self.print_stats:
            print("Mean search time", mean_search_time_correct)
            print("Mean ratio correct:", mean_ratio_correct)
            print("Average program length", avg_program_length)

        return {"Mean search time": mean_search_time_correct,
                "Mean ratio correct:": mean_ratio_correct,
                "Average program length": avg_program_length,
                "Domain-specific language": genome}


def sort_genome(genome: Genome) -> Genome:
    sorted_bool = sorted(genome.get_bool_tokens(), key=str, reverse=False)
    sorted_trans = sorted(genome.get_trans_tokens(), key=str, reverse=False)

    return DomainSpecificLanguage(genome.domain_name, sorted_bool, sorted_trans)


def get_fitness(genome: Genome):
    return genome_fitness_values[str(genome)]["fitness"]


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


def crossover_exchange_halve_category(a: Genome, b: Genome) -> Tuple[Genome, Genome]:
    # Select half of a bool and trans, keep track of remainders
    # Select half of b bool and trans, keep track of remainders

    # Merge halves, and remainders

    a_bool_selected, a_bool_remainder = get_random_half_and_remainder(a.get_bool_tokens())
    a_trans_selected, a_trans_remainder = get_random_half_and_remainder(a.get_trans_tokens())

    b_bool_selected, b_bool_remainder = get_random_half_and_remainder(b.get_bool_tokens())
    b_trans_selected, b_trans_remainder = get_random_half_and_remainder(b.get_trans_tokens())

    crossed_a = sort_genome(DomainSpecificLanguage(a.domain_name, list(set(a_bool_selected + b_bool_selected)),
                                                   list(set(a_trans_selected + b_trans_selected))))
    crossed_b = sort_genome(DomainSpecificLanguage(a.domain_name, list(set(a_bool_remainder + b_bool_remainder)),
                                                   list(set(a_trans_remainder + b_trans_remainder))))

    return crossed_a, crossed_b


def crossover_exchange_halve_random(a: Genome, b: Genome) -> Tuple[Genome, Genome]:
    # Select half of a, keep track of remainder
    # Select half of b, keep track of remainder

    a_tokens = a.get_bool_tokens() + a.get_trans_tokens()
    b_tokens = b.get_bool_tokens() + b.get_trans_tokens()

    a_tokens_selected, a_tokens_remainder = get_random_half_and_remainder(a_tokens)
    b_tokens_selected, b_tokens_remainder = get_random_half_and_remainder(b_tokens)

    selected_no_duplicates = remove_duplicates(a_tokens_selected + b_tokens_selected)
    remainder_no_duplicates = remove_duplicates(a_tokens_remainder + b_tokens_remainder)

    return create_dsl_from_tokens(a.domain_name, selected_no_duplicates), \
           create_dsl_from_tokens(a.domain_name, remainder_no_duplicates)


# Created because we can't do set conversion for some tokens
def remove_duplicates(tokens: List[Token]) -> List[Token]:
    final_tokens = []
    for token in tokens:
        if token not in final_tokens:
            final_tokens.append(token)
    return final_tokens


def get_random_half_and_remainder(tokens: List[Token]) -> Tuple[List[Token], List[Token]]:
    tokens_selected = random.sample(tokens, round(len(tokens) / 2))
    tokens_remainder = [v for v in tokens if v not in tokens_selected]
    return tokens_selected, tokens_remainder


def create_dsl_from_tokens(domain: str, tokens: List[Token]) -> Genome:
    bool_tokens = []
    trans_tokens = []

    dsl = StandardDomainSpecificLanguage(domain)
    for token in tokens:
        if token in dsl.get_bool_tokens():
            bool_tokens.append(token)
        else:
            trans_tokens.append(token)

    return sort_genome(DomainSpecificLanguage(dsl.domain_name, bool_tokens, trans_tokens))


# MUTATION FUNCTIONS
def mutate_add_token(genome: Genome, dsl: DomainSpecificLanguage) -> Genome:
    bool_tokens_genome = genome.get_bool_tokens()
    trans_tokens_genome = genome.get_trans_tokens()

    all_tokens_dsl = dsl.get_bool_tokens() + dsl.get_trans_tokens()
    all_tokens_genome = bool_tokens_genome + trans_tokens_genome

    optional_tokens = []
    for token in all_tokens_dsl:
        if token not in all_tokens_genome:
            optional_tokens.append(token)

    if len(optional_tokens) == 0:
        return DomainSpecificLanguage(genome.domain_name, bool_tokens_genome, trans_tokens_genome)

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


def mutate_add_special_token(genome: Genome) -> Genome:
    bool_tokens_genome = genome.get_bool_tokens()
    trans_tokens_genome = genome.get_trans_tokens()

    randomly_selected_token = pick_random_weighted()

    if randomly_selected_token not in trans_tokens_genome:
        trans_tokens_genome.append(randomly_selected_token)

    result = sort_genome(DomainSpecificLanguage(genome.domain_name, bool_tokens_genome, trans_tokens_genome))
    return result


def pick_random_weighted():
    rand_val = random.random()
    total = 0
    for k, v in successful_tokens_weights.items():
        total += v
        if rand_val <= total:
            return successful_tokens_objects[k]
    return None


def normalize_special_token_weights():
    # Since we want to enlarge the chance that various tokens get picked, we make smaller weights relatively bigger
    for key in successful_tokens_counts:
        successful_tokens_weights[key] = (successful_tokens_counts[key] ** 0.2)

    total_token_count = sum(successful_tokens_counts.values())

    for key in successful_tokens_counts:
        successful_tokens_weights[key] = successful_tokens_counts[key] / total_token_count
        # print(str(key), successful_tokens_weights[key])


def remove_specific_tokens(genome: Genome, tokens: List[Token]):
    bool_tokens_genome = genome.get_bool_tokens()
    trans_tokens_genome = genome.get_trans_tokens()

    for token in tokens:
        if token in bool_tokens_genome:
            bool_tokens_genome.remove(token)
        elif token in trans_tokens_genome:
            trans_tokens_genome.remove(token)

    result = sort_genome(DomainSpecificLanguage(genome.domain_name, bool_tokens_genome, trans_tokens_genome))
    return result


def process_search_results(search_results: dict, domain: str) -> Tuple[float, float, List]:
    total_cases = 0
    cumulative_ratios_correct = 0
    cumulative_search_time = 0

    best_programs = []

    for key, value in search_results.items():
        total_cases += 1
        current_search_result = value
        search_time = current_search_result["search_time"]
        test_total = current_search_result["test_total"]
        best_programs.append(current_search_result["best_program"])

        current_ratio_correct = 0
        # If we use string domain we should use tests, instead of training examples
        if domain == "string":
            test_correct = current_search_result["test_correct"]
            current_ratio_correct += test_correct / test_total
        else:
            train_correct = current_search_result["train_correct"]
            current_ratio_correct += train_correct / test_total

        cumulative_search_time += search_time
        cumulative_ratios_correct += current_ratio_correct

    mean_ratio_correct = cumulative_ratios_correct / total_cases
    mean_search_time_correct = cumulative_search_time / total_cases

    return mean_ratio_correct, mean_search_time_correct, best_programs


def extract_special_tokens(best_programs: List, dsl: DomainSpecificLanguage):
    for program in best_programs:
        tokens = program.sequence

        for token in tokens:
            if token not in dsl.get_trans_tokens():

                if str(token) not in successful_tokens_counts.keys():
                    successful_tokens_counts[str(token)] = 1
                    successful_tokens_objects[str(token)] = token
                else:
                    successful_tokens_counts[str(token)] = successful_tokens_counts[str(token)] + 1

                # print(str(token), successful_tokens_counts[str(token)])


def get_chromosome_stats(chromosome: DomainSpecificLanguage):
    stats = genome_fitness_values[str(chromosome)]
    return {"correct": stats["correct"],
            "search": stats["search"],
            "fitness": stats["fitness"]}


def print_chromosome_stats(chromosome: DomainSpecificLanguage):
    stats = genome_fitness_values[str(chromosome)]
    print("correct:", round(stats["correct"], 3),
          "search:", round(stats["search"], 3),
          "fitness:", round(stats["fitness"], 3),
          str(stats["genome"]))
