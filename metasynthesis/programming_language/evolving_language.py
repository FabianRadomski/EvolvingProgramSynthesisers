import copy
import random
from random import randrange
from typing import List, Callable, Tuple, Iterable

from common.program_synthesis.dsl import DomainSpecificLanguage, StandardDomainSpecificLanguage
from common.tokens.abstract_tokens import Token, InventedToken, ControlToken
from common.tokens.control_tokens import LoopWhile, If
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
        best_programs = []

        for key, value in search_results.items():
            total_cases += 1
            current_search_result = value
            search_time = current_search_result["search_time"]
            test_total = current_search_result["test_total"]
            best_programs.append(current_search_result["best_program"])


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


        # SPECIAL TOKEN EXTRACTION
        # total_search_time = 0

        for program in best_programs:
            tokens = program.sequence

            # total_search_time += result.dictionary["execution_time"]

            for token in tokens:
                if token not in self.trans_tokens:

                    if str(token) not in successful_tokens_counts.keys():
                        successful_tokens_counts[str(token)] = 1
                        successful_tokens_objects[str(token)] = token
                    else:
                        successful_tokens_counts[str(token)] = successful_tokens_counts[str(token)] + 1

                    # print(str(token), successful_tokens_counts[str(token)])

        genome_fitness_values[str(genome)] = fitness_value

        return fitness_value

    def crossover(self, a: Genome, b: Genome) -> Tuple[Genome, Genome]:
        """This method applies the given crossover function with certain probability"""

        rand_float = random.random()
        crossed_a = None
        crossed_b = None

        if 0 <= rand_float <= 0:
            crossed_a, crossed_b = crossover_exchange_trans_bool(a, b)
        elif 0 < rand_float <= 0:
            crossed_a, crossed_b = crossover_exchange_halve_category(a, b)
        elif 0 < rand_float <= 1:
            crossed_a, crossed_b = crossover_exchange_halve_random(a, b)

        return crossed_a, crossed_b

    def mutation(self, genome: Genome) -> Genome:
        """This method applies mutation to a genome with certain probability"""

        rand_float = random.random()
        mutated_genome = genome

        if 0 <= rand_float <= 0.45:
            mutated_genome = mutate_add_special_token(genome)
        elif 0.45 < rand_float <= 0.5:
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
            successful_tokens_weights.clear()

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

            # SPECIAL TOKEN WEIGHTS
            normalize_token_weights()

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

    selected_unduplicated = remove_duplicates(a_tokens_selected + b_tokens_selected)
    remainder_unduplicated = remove_duplicates(a_tokens_remainder + b_tokens_remainder)

    return create_dsl_from_tokens(a.domain_name, selected_unduplicated), \
           create_dsl_from_tokens(a.domain_name, remainder_unduplicated)

# Had to be introduced because we can't do set conversion for some tokens
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


def mutate_add_special_token(genome: Genome) -> Genome:
    bool_tokens_genome = genome.get_bool_tokens()
    trans_tokens_genome = genome.get_trans_tokens()

    randomly_selected_token = pick_random_weighted()
    # print(type(randomly_selected_token), str(randomly_selected_token))
    # print(str(flatten_token(randomly_selected_token)))

    # Random chance of deleting the tokens that are already in the special token
    rand_float = random.random()
    if rand_float < 0.2:
        genome = remove_specific_tokens(genome, list(flatten_token(randomly_selected_token)))

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


def normalize_token_weights():

    # Since we want enlarge the chance that various tokens get picked, we make smaller weights relatively bigger
    for key in successful_tokens_counts:
        successful_tokens_weights[key] = (successful_tokens_counts[key] ** 0.1)

    total_token_count = sum(successful_tokens_counts.values())

    for key in successful_tokens_counts:
        successful_tokens_weights[key] = successful_tokens_counts[key] / total_token_count
        # print(str(key), successful_tokens_weights[key])


def flatten_token(token: Token):
    flat_token_set = set()

    if type(token) is InventedToken:
        base_tokens = token.tokens
        for each_token in base_tokens:
            flat_token_set.update(flatten_token(each_token))
    elif type(token) is If:
        flat_token_set.update(flatten_token(token.cond))
        flat_token_set.update(flatten_token(token.e1))
        flat_token_set.update(flatten_token(token.e2))
    elif type(token) is LoopWhile:
        flat_token_set.update(flatten_token(token.cond))
        for body_token in token.loop_body:
            flat_token_set.update(flatten_token(body_token))
    elif type(token) is List:
        for t in token:
            flat_token_set.update(flatten_token(t))
    else:
        flat_token_set.add(token)

    return flat_token_set


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
