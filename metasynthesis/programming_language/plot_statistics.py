from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np

from common.program_synthesis.dsl import StandardDomainSpecificLanguage
from metasynthesis.programming_language.evolve_language import EvolvingLanguage


class LanguageStatistics:

    def __init__(self, print_stats: bool = True, search_mode: str = "debug"):
        self.print_stats = print_stats
        self.search_mode = search_mode
        self.base_save_path = "metasynthesis/programming_language/results/" + self.search_mode + "/"

        plt.style.use('seaborn-colorblind')  # ggplot
        # plt.style.use('tableau-colorblind10')

        # These parameters are manually updated based on analysis of the plots
        self.best_parameters = {"generation_limit": 50,
                                "generation_size": 30,
                                "search_mode": search_mode,
                                "search_algorithm": "AS",
                                "search_setting": "SO",
                                "max_search_time": 2,
                                "crossover_probability": 0.8,
                                "mutation_probability": 0.2
                                }

    def plot_search_algorithm_performance(self):

        search_algorithms = ["Brute", "AS", "LNS", "MH"]

        general_genetic = EvolvingLanguage(max_search_time=self.best_parameters["max_search_time"],
                                           generation_size=self.best_parameters["generation_size"])
        start_population = general_genetic.generate_population()

        # final_ratios_correct = []
        # cumulative_times = []

        for algorithm in search_algorithms:
            if self.print_stats:
                print("SEARCH ALGORITHM:", algorithm)

            genetic = EvolvingLanguage(generation_limit=self.best_parameters["generation_limit"],
                                       generation_size=self.best_parameters["generation_size"],
                                       search_mode=self.best_parameters["search_mode"],
                                       max_search_time=self.best_parameters["max_search_time"],
                                       crossover_probability=self.best_parameters["crossover_probability"],
                                       mutation_probability=self.best_parameters["mutation_probability"],
                                       start_population=start_population,
                                       search_algo=algorithm)
            all_results = genetic.run_evolution()

            _, generation_times, average_fitness_values = \
                extract_generation_time_and_fitness(all_results, "Best fitness")

            # cumulative_generation_times = [sum(generation_times[0:i[0]]) for i in enumerate(generation_times)]

            final_ratio_correct = all_results["Final evolved results"]["Mean ratio correct"]
            cumulative_time = sum(generation_times)

            plt.plot(cumulative_time, final_ratio_correct, marker="o", markersize=14, label=algorithm)

            # plt.plot(cumulative_generation_times, average_fitness_values, label=algorithm)
        #
        # plt.xlabel("Total time passed")
        # plt.ylabel("Best chromosome fitness")
        # # plt.title("Effect of search algorithm on time and fitness")
        # plt.legend(loc="upper right")
        # plt.savefig(self.base_save_path + "search_algorithm_comparison.jpg")
        # plt.close()

        plt.xlabel("Total time passed (s)")
        plt.ylabel("Evolved chromosome correct ratio")
        # plt.title("Effect of search algorithm on time and fitness")
        plt.legend(loc="upper right")
        plt.savefig(self.base_save_path + "search_algorithm_comparison_dots.jpg")
        plt.close()

    def plot_search_setting_performance(self):
        search_settings = []
        domain = "string"
        if domain == "string":
            search_settings = ["SG", "SO", "SE"]
        elif domain == "robot":
            search_settings = ["RG", "RO", "RE"]
        elif domain == "pixel":
            search_settings = ["PG", "PO", "PE"]

        general_genetic = EvolvingLanguage(max_search_time=self.best_parameters["max_search_time"],
                                           generation_size=self.best_parameters["generation_size"])
        start_population = general_genetic.generate_population()

        for setting in search_settings:
            if self.print_stats:
                print("SEARCH SETTING:", setting)

            genetic = EvolvingLanguage(generation_limit=self.best_parameters["generation_limit"],
                                       generation_size=self.best_parameters["generation_size"],
                                       search_mode=self.best_parameters["search_mode"],
                                       max_search_time=self.best_parameters["max_search_time"],
                                       crossover_probability=self.best_parameters["crossover_probability"],
                                       mutation_probability=self.best_parameters["mutation_probability"],
                                       start_population=start_population,
                                       search_algo=self.best_parameters["search_algorithm"],
                                       search_setting=setting)
            all_results = genetic.run_evolution()

            generations, generation_times, average_fitness_values = \
                extract_generation_time_and_fitness(all_results, "Best fitness")

            # cumulative_generation_times = [sum(generation_times[0:i[0]]) for i in enumerate(generation_times)]
            generations_stats = all_results["Generation statistics"]
            generations_best_correct = []

            for gen in generations_stats:
                generations_best_correct.append(generations_stats[gen]["Generation best stats"]["correct"])

            plt.plot(generations, generations_best_correct, label=setting)

        plt.xlabel("Generations")
        plt.ylabel("Ratio correct best chromosome")
        # plt.title("Effect of search setting on time and fitness")
        plt.legend(loc="upper left")
        plt.savefig(self.base_save_path + "search_setting_comparison.jpg")
        plt.close()

    def plot_search_timeout_performance(self):
        timeouts = [0.1, 0.5, 1, 2, 5, 10]

        general_genetic = EvolvingLanguage(max_search_time=self.best_parameters["max_search_time"],
                                           generation_size=self.best_parameters["generation_size"])
        start_population = general_genetic.generate_population()

        for timeout in timeouts:
            if self.print_stats:
                print("TIMEOUT:", timeout)

            genetic = EvolvingLanguage(generation_limit=self.best_parameters["generation_limit"],
                                       generation_size=self.best_parameters["generation_size"],
                                       search_mode=self.best_parameters["search_mode"],
                                       max_search_time=timeout,
                                       crossover_probability=self.best_parameters["crossover_probability"],
                                       mutation_probability=self.best_parameters["mutation_probability"],
                                       start_population=start_population,
                                       search_algo=self.best_parameters["search_algorithm"],
                                       search_setting=self.best_parameters["search_setting"])
            all_results = genetic.run_evolution()

            generations, generation_times, average_fitness_values = \
                extract_generation_time_and_fitness(all_results, "Average fitness")

            generations_stats = all_results["Generation statistics"]
            generations_best_correct = []

            for gen in generations_stats:
                generations_best_correct.append(generations_stats[gen]["Generation best stats"]["correct"])

            plt.figure(1)
            plt.plot(generations, generations_best_correct, label=timeout)

            plt.figure(2)
            plt.plot(generations, generation_times, label=timeout)

        plt.figure(1)
        plt.xlabel("Generation")
        plt.ylabel("Ratio correct best chromosome")
        # plt.title("Effect of timeout on ratio correct")
        plt.legend(loc="lower right")
        plt.savefig(self.base_save_path + "search_timeout_comparison_correct.jpg")

        plt.figure(2)
        plt.xlabel("Generation")
        plt.ylabel("Generation time taken (s)")
        # plt.title("Effect of timeout on generation time")
        plt.legend(loc="upper right")
        plt.savefig(self.base_save_path + "search_timeout_comparison_time.jpg")

        plt.close('all')

    def plot_mutation_method_performance(self):
        # Add_construct, Add_random, Remove_random,
        mutation_params = [[0, 1 / 2, 1 / 2], [1 / 2, 0, 1 / 2], [1 / 4, 1 / 4, 1 / 2], [2 / 5, 1 / 10, 1 / 2]]

        general_genetic = EvolvingLanguage(max_search_time=self.best_parameters["max_search_time"],
                                           generation_size=10)

        # Other important parameters
        cross_prob = 0
        mutation_prob = 0.3
        start_population = general_genetic.generate_population()

        for weights in mutation_params:
            if self.print_stats:
                print("MUTATION WEIGHTS:", weights)

            genetic = EvolvingLanguage(generation_limit=60,
                                       generation_size=20,
                                       search_mode=self.best_parameters["search_mode"],
                                       max_search_time=self.best_parameters["max_search_time"],
                                       crossover_probability=cross_prob,
                                       mutation_probability=mutation_prob,
                                       mutation_weights=weights,
                                       start_population=start_population)
            all_results = genetic.run_evolution()

            num_explored_languages = all_results["Explored languages"]
            generations, _, average_fitness_values = extract_generation_time_and_fitness(all_results, "Best fitness")
            plt.plot(generations, average_fitness_values, label="weights:" + str(weights))

        plt.xlabel("Generations")
        plt.ylabel("Best chromosome fitness")
        # plt.title("Effect of mutation methods on fitness")
        plt.legend(loc="lower right")
        plt.savefig(self.base_save_path + "mutation_method_comparison.jpg")
        plt.close()

    def plot_crossover_method_performance(self):
        # Trans_bool, Half_each, Half_random, mixed
        crossover_params = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.33, 0.33, 0.33]]

        general_genetic = EvolvingLanguage(max_search_time=self.best_parameters["max_search_time"],
                                           generation_size=self.best_parameters["generation_size"])

        # Other important parameters
        cross_prob = 1
        mutation_prob = 0
        start_population = general_genetic.generate_population()

        for weights in crossover_params:
            if self.print_stats:
                print("CROSSOVER WEIGHTS:", weights)

            genetic = EvolvingLanguage(generation_limit=self.best_parameters["generation_limit"],
                                       generation_size=self.best_parameters["generation_size"],
                                       search_mode=self.best_parameters["search_mode"],
                                       max_search_time=self.best_parameters["max_search_time"],
                                       crossover_probability=cross_prob,
                                       mutation_probability=mutation_prob,
                                       crossover_weights=weights,
                                       start_population=start_population)
            all_results = genetic.run_evolution()

            num_explored_languages = all_results["Explored languages"]
            generations, _, average_fitness_values = extract_generation_time_and_fitness(all_results, "Best fitness")
            plt.plot(generations, average_fitness_values, label="weights:" + str(weights))

        plt.xlabel("Generations")
        plt.ylabel("Best chromosome fitness")
        # plt.title("Effect of crossover methods on fitness")
        plt.legend(loc="upper left")
        plt.savefig(self.base_save_path + "crossover_method_comparison.jpg")
        plt.close()

    def plot_population_size_performance(self):
        generation_sizes = [10, 20, 30, 40]

        # Other important parameters
        generation_limit = 30

        for size in generation_sizes:
            if self.print_stats:
                print("POPULATION SIZE:", size)

            genetic = EvolvingLanguage(generation_limit=generation_limit,
                                       generation_size=size,
                                       search_mode=self.best_parameters["search_mode"],
                                       max_search_time=self.best_parameters["max_search_time"],
                                       crossover_probability=self.best_parameters["crossover_probability"],
                                       mutation_probability=self.best_parameters["mutation_probability"])
            all_results = genetic.run_evolution()

            num_explored_languages = all_results["Explored languages"]
            generations, generation_times, best_fitness_values = extract_generation_time_and_fitness(all_results,
                                                                                                     "Best fitness")

            cumulative_generation_times = [sum(generation_times[0:i[0]]) for i in enumerate(generation_times)]

            plt.plot(cumulative_generation_times, best_fitness_values, label=str(size))

        plt.xlabel("Total time passed")
        plt.ylabel("Best chromosome fitness")
        # plt.title("Effect of population size on fitness vs time passed")
        plt.legend(loc="upper left")
        plt.savefig(self.base_save_path + "population_size_comparison.jpg")
        plt.close()

    def plot_generation_limit_performance(self):

        cumulative_fitness_values = [0] * 50
        generations = []
        for i in range(0, 5):
            if self.print_stats:
                print("GENERATION LIMIT:")

            genetic = EvolvingLanguage(generation_limit=50,
                                       generation_size=self.best_parameters["generation_size"],
                                       search_mode=self.best_parameters["search_mode"],
                                       max_search_time=self.best_parameters["max_search_time"],
                                       crossover_probability=self.best_parameters["crossover_probability"],
                                       mutation_probability=self.best_parameters["mutation_probability"])
            all_results = genetic.run_evolution()

            num_explored_languages = all_results["Explored languages"]
            generations, generation_times, best_fitness_values = extract_generation_time_and_fitness(all_results,
                                                                                                     "Best fitness")
            for i, fit in enumerate(best_fitness_values):
                cumulative_fitness_values[i] += fit
            # cumulative_generation_times = [sum(generation_times[0:i[0]]) for i in enumerate(generation_times)]

        avg_fitness_values = [0] * 50
        for i, fit in enumerate(cumulative_fitness_values):
            avg_fitness_values[i] += cumulative_fitness_values[i] / 5

        plt.plot(generations, avg_fitness_values, label="50 generations")

        plt.xlabel("Generations")
        plt.ylabel("Best chromosome fitness")
        # plt.title("Effect of generation limit on fitness vs time passed")
        plt.legend(loc="lower right")
        plt.savefig(self.base_save_path + "generation_limit_comparison.jpg")
        plt.close()

    def plot_standard_vs_evolved_performance(self):
        domains = ["string", "robot", "pixel"]  # ", "pixel"
        left_bound_cases = 0
        right_bound_cases = 1

        final_standard_corrects = []
        final_evolved_corrects = []

        final_standard_search_time = []
        final_evolved_search_time = []

        final_standard_program_length = []
        final_evolved_program_length = []

        for domain in domains:
            if self.print_stats:
                print("FINAL DOMAIN:", domain)

            if domain == "string":
                search_setting = "SO"
            elif domain == "robot":
                search_setting = "RO"
            else:
                search_setting = "PO"

            genetic = EvolvingLanguage(generation_limit=self.best_parameters["generation_limit"],
                                       generation_size=self.best_parameters["generation_size"],
                                       search_algo=self.best_parameters["search_algorithm"],
                                       search_mode="eval_train",  # self.best_parameters["search_mode"],
                                       search_setting=search_setting,
                                       max_search_time=self.best_parameters["max_search_time"],
                                       crossover_probability=self.best_parameters["crossover_probability"],
                                       mutation_probability=self.best_parameters["mutation_probability"],
                                       dsl=StandardDomainSpecificLanguage(domain),
                                       left_bound_cases=left_bound_cases,
                                       right_bound_cases=right_bound_cases)
            all_results = genetic.run_evolution()

            final_standard_results = all_results["Final original results"]
            final_evolved_results = all_results["Final evolved results"]

            final_standard_corrects.append(final_standard_results["Mean ratio correct"])
            final_evolved_corrects.append(final_evolved_results["Mean ratio correct"])

            final_standard_search_time.append(final_standard_results["Mean search time"])
            final_evolved_search_time.append(final_evolved_results["Mean search time"])

            final_standard_program_length.append(final_standard_results["Average program length"])
            final_evolved_program_length.append(final_evolved_results["Average program length"])

        final_standard_corrects = convert_to_percentage(final_standard_corrects)
        final_evolved_corrects = convert_to_percentage(final_evolved_corrects)

        y = np.arange(len(domains))
        size = 0.3

        fig, ax = plt.subplots()
        rects1 = ax.barh(y - size / 2, final_standard_corrects, size, label='Standard language')
        rects2 = ax.barh(y + size / 2, final_evolved_corrects, size, label='Evolved language')

        ax.set_xlabel("Percentage solved tasks", fontsize=14)
        ax.set_yticks(y, domains, fontsize=14)
        ax.invert_yaxis()
        ax.legend(loc='lower right', fontsize=12)

        ax.bar_label(rects1, convert_percentage_label(round_list(final_standard_corrects)), padding=3, fontsize=12)
        ax.bar_label(rects2, convert_to_percentage_diff(final_standard_corrects, final_evolved_corrects), padding=3,
                     fontsize=12)
        ax.set_xlim(right=119)
        ax.xaxis.grid(True)
        fig.tight_layout()
        plt.savefig(self.base_save_path + "final/" + str(left_bound_cases) + "-" + str(
            right_bound_cases) + "_ratio_correct_comparison.jpg")
        plt.close()

        fig, ax = plt.subplots()
        rects1 = ax.barh(y - size / 2, final_standard_search_time, size, label='Standard language')
        rects2 = ax.barh(y + size / 2, final_evolved_search_time, size, label='Evolved language')

        ax.set_xlabel("Average search time (s)", fontsize=14)
        ax.set_yticks(y, domains, fontsize=14)
        ax.invert_yaxis()
        ax.legend(loc='center right', fontsize=12)

        ax.bar_label(rects1, round_list(final_standard_search_time), padding=3, fontsize=12)
        ax.bar_label(rects2, convert_to_percentage_diff(final_standard_search_time, final_evolved_search_time),
                     padding=3,
                     fontsize=12)
        ax.set_xlim(right=2)
        ax.xaxis.grid(True)
        fig.tight_layout()
        plt.savefig(self.base_save_path + "final/" + str(left_bound_cases) + "-" + str(
            right_bound_cases) + "_search_time_comparison.jpg")

        plt.close()

        fig, ax = plt.subplots()
        rects1 = ax.barh(y - size / 2, final_standard_program_length, size, label='Standard language')
        rects2 = ax.barh(y + size / 2, final_evolved_program_length, size, label='Evolved language')

        ax.set_xlabel("Average program length", fontsize=14)
        ax.set_yticks(y, domains, fontsize=14)
        ax.invert_yaxis()
        ax.legend(loc='upper right', fontsize=12)

        ax.bar_label(rects1, round_list(final_standard_program_length), padding=3, fontsize=12)
        ax.bar_label(rects2, convert_to_percentage_diff(final_standard_program_length, final_evolved_program_length),
                     padding=3,
                     fontsize=12)
        ax.set_xlim(right=35)
        ax.xaxis.grid(True)
        fig.tight_layout()
        plt.savefig(self.base_save_path + "final/" + str(left_bound_cases) + "-" + str(
            right_bound_cases) + "_program_length_comparison.jpg")

        plt.close()

        #
        # plt.bar(X_axis - 0.2, final_standard_program_length, 0.4, label='Standard language')
        # plt.bar(X_axis + 0.2, final_evolved_program_length, 0.4, label='Evolved language')
        #
        # plt.xticks(X_axis, domains)
        # plt.xlabel("Domains")
        # plt.ylabel("Program length")
        # plt.legend(loc="upper left")
        # plt.savefig(self.base_save_path + "final/" + str(left_bound_cases) + "-" + str(right_bound_cases) + "program_length_comparison.jpg")
        # plt.close()


def round_list(ls: List):
    r = []
    for x in ls:
        r.append(round(x, 2))
    return r


def convert_to_percentage(corrects: List):
    corrects_percentages = []
    for x in corrects:
        corrects_percentages.append(x * 100)
    return corrects_percentages


def convert_percentage_label(percentages: List):
    labels = []
    for x in percentages:
        labels.append(str(x) + "%")
    return labels


def convert_to_percentage_diff(standard: List, evolved: List):
    labels = []
    for i, val in enumerate(standard):
        if val != 0:
            perc = (evolved[i] - val) / val
        else:
            perc = 0
        perc *= 100

        perc = round(perc, 1)

        if perc >= 0:
            perc = "+" + str(perc) + "%"
        else:
            perc = str(perc) + "%"
        labels.append(perc)

    return labels


def extract_generation_time_and_fitness(all_results: Dict, fitness_to_get: str):
    generation_statistics = all_results["Generation statistics"]

    generations = []
    generation_times = []
    average_fitness_values = []
    for gen in generation_statistics.keys():
        generations.append(float(gen))
        generation_times.append(generation_statistics[gen]["Time taken"])
        average_fitness_values.append(generation_statistics[gen][fitness_to_get])

    return generations, generation_times, average_fitness_values


def simple_plot_with_params(xs: List, ys: List, xlabel: str, ylabel: str, title: str):
    plt.plot(xs, ys)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.close()
