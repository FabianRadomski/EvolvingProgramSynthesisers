from typing import List, Dict

import matplotlib.pyplot as plt

from common.program_synthesis.dsl import StandardDomainSpecificLanguage
from metasynthesis.programming_language.evolve_language import EvolvingLanguage


class LanguageStatistics:

    def __init__(self, domain: str):
        self.domain = domain
        self.dsl = StandardDomainSpecificLanguage(domain)

        # These parameters are manually updated based on analysis of the plots
        self.best_parameters = {"generation_limit": 2,
                                "generation_size": 6,
                                "search_mode": "debug",
                                "search_algorithm": "AS",
                                "max_search_time": 1,
                                "crossover_probability": 0.8,
                                "mutation_probability": 0.3,
                                }

    def plot_search_algorithm_performance(self):
        plt.style.use('seaborn-paper')  # ggplot

        search_algorithms = ["Brute", "AS", "LNS", "MH"]
        # search_algorithms = ["LNS"]


        general_genetic = EvolvingLanguage(max_search_time=self.best_parameters["max_search_time"],
                                           generation_size=self.best_parameters["generation_size"])
        start_population = general_genetic.generate_population()

        for algorithm in search_algorithms:
            genetic = EvolvingLanguage(generation_limit=self.best_parameters["generation_limit"],
                                       generation_size=self.best_parameters["generation_size"],
                                       search_mode=self.best_parameters["search_mode"],
                                       max_search_time=self.best_parameters["max_search_time"],
                                       crossover_probability=self.best_parameters["crossover_probability"],
                                       mutation_probability=self.best_parameters["mutation_probability"],
                                       dsl=self.dsl,
                                       start_population=start_population,
                                       search_algo=algorithm)
            all_results = genetic.run_evolution()

            _, generation_times, average_fitness_values = \
                extract_generation_time_and_fitness(all_results, "Average fitness")

            plt.plot(generation_times, average_fitness_values, label=algorithm)

            # time_taken = all_results["Total time taken"]
            # num_explored_languages = all_results["Explored languages"]
            #
            # plt.scatter(time_taken, num_explored_languages, label=algorithm)

        plt.xlabel("Time taken per generation")
        plt.ylabel("Average generation fitness")
        plt.title("Effect of search setting on generation time and fitness")
        plt.legend(loc="upper left")
        plt.savefig("metasynthesis/programming_language/results/search_algorithm_comparison.jpg")
        plt.show()

    def plot_search_setting_performance(self):
        search_settings = []
        if self.domain == "string":
            search_settings = ["SG", "SO", "SE"]
        elif self.domain == "robot":
            search_settings = ["RG", "RO", "RE"]
        elif self.domain == "pixel":
            search_settings = ["RG", "RO", "RE"]

        general_genetic = EvolvingLanguage(max_search_time=self.best_parameters["max_search_time"],
                                           generation_size=self.best_parameters["generation_size"])
        start_population = general_genetic.generate_population()

        for setting in search_settings:
            genetic = EvolvingLanguage(generation_limit=self.best_parameters["generation_limit"],
                                       generation_size=self.best_parameters["generation_size"],
                                       search_mode=self.best_parameters["search_mode"],
                                       max_search_time=self.best_parameters["max_search_time"],
                                       crossover_probability=self.best_parameters["crossover_probability"],
                                       mutation_probability=self.best_parameters["mutation_probability"],
                                       dsl=self.dsl,
                                       start_population=start_population,
                                       search_algo=self.best_parameters["search_algorithm"],
                                       search_setting=setting)
            all_results = genetic.run_evolution()

            _, generation_times, average_fitness_values = \
                extract_generation_time_and_fitness(all_results, "Average fitness")

            plt.plot(generation_times, average_fitness_values, label=setting)

        plt.xlabel("Time taken per generation")
        plt.ylabel("Average generation fitness")
        plt.title("Effect of search setting on generation time and fitness")
        plt.legend(loc="upper left")
        plt.savefig("metasynthesis/programming_language/results/search_setting_comparison.jpg")
        plt.show()

    def plot_mutation_method_performance(self):
        # Add_construct, Add_random, Remove_random,
        mutation_params = [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.25, 0.25, 0.5],
                           [0.4, 0.1, 0.5], [1, 0, 0], [0, 1, 0], [0, 0, 1]]

        general_genetic = EvolvingLanguage(max_search_time=self.best_parameters["max_search_time"],
                                           generation_size=self.best_parameters["generation_size"])

        # Other important parameters
        cross_prob = 0
        mutation_prob = 0.3
        start_population = general_genetic.generate_population()

        for weights in mutation_params:
            genetic = EvolvingLanguage(generation_limit=self.best_parameters["generation_limit"],
                                       generation_size=self.best_parameters["generation_size"],
                                       search_mode=self.best_parameters["search_mode"],
                                       max_search_time=self.best_parameters["max_search_time"],
                                       crossover_probability=cross_prob,
                                       mutation_probability=mutation_prob,
                                       dsl=self.dsl,
                                       mutation_weights=weights,
                                       start_population=start_population)
            all_results = genetic.run_evolution()

            num_explored_languages = all_results["Explored languages"]
            generations, _, average_fitness_values = extract_generation_time_and_fitness(all_results, "Best fitness")
            plt.plot(generations, average_fitness_values, label="weights:" + str(weights)
                                                                + " explored:" + str(num_explored_languages))

        plt.xlabel("Generations")
        plt.ylabel("Best fitness")
        plt.title("Effect of mutation methods on fitness")
        plt.legend(loc="upper left")
        plt.savefig("metasynthesis/programming_language/results/mutation_method_comparison.jpg")
        plt.show()

    def plot_crossover_method_performance(self):
        # Trans_bool, Half_each, Half_random,
        crossover_params = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        general_genetic = EvolvingLanguage(max_search_time=self.best_parameters["max_search_time"],
                                           generation_size=self.best_parameters["generation_size"])

        # Other important parameters
        cross_prob = 1
        mutation_prob = 0
        start_population = general_genetic.generate_population()

        for weights in crossover_params:
            genetic = EvolvingLanguage(generation_limit=self.best_parameters["generation_limit"],
                                       generation_size=self.best_parameters["generation_size"],
                                       search_mode=self.best_parameters["search_mode"],
                                       max_search_time=self.best_parameters["max_search_time"],
                                       crossover_probability=cross_prob,
                                       mutation_probability=mutation_prob,
                                       dsl=self.dsl,
                                       crossover_weights=weights,
                                       start_population=start_population)
            all_results = genetic.run_evolution()

            num_explored_languages = all_results["Explored languages"]
            generations, _, average_fitness_values = extract_generation_time_and_fitness(all_results, "Average fitness")
            plt.plot(generations, average_fitness_values, label="weights:" + str(weights)
                                                                + " explored:" + str(num_explored_languages))

        plt.xlabel("Generations")
        plt.ylabel("Average fitness")
        plt.title("Effect of crossover methods on fitness")
        plt.legend(loc="upper left")
        plt.savefig("metasynthesis/programming_language/results/crossover_method_comparison.jpg")
        plt.show()

    def plot_search_time_performance(self):
        pass


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
    plt.show()
