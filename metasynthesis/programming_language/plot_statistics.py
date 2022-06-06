from typing import List, Dict

import matplotlib.pyplot as plt

from common.program_synthesis.dsl import StandardDomainSpecificLanguage
from metasynthesis.programming_language.evolve_language import EvolvingLanguage


class LanguageStatistics:

    def __init__(self, domain: str):
        self.domain = domain
        self.dsl = StandardDomainSpecificLanguage(domain)

        # These parameters are manually updated based on analysis of the plots
        self.best_parameters = {"generation_limit": 3,
                                "generation_size": 4,
                                "search_mode": "debug",
                                "search_algorithm": "AS",
                                "search_setting": "SG",
                                "max_search_time": 0.01,
                                "crossover_probability": 0.8,
                                "mutation_probability": 0.3,
                                }

    def plot_search_algorithm_performance(self):
        plt.style.use('ggplot')  # ggplot

        # search_algorithms = ["Brute", "AS", "LNS", "MH"]
        search_algorithms = ["Brute"]

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

            cumulative_generation_times = [sum(generation_times[0:i[0]]) for i in enumerate(generation_times)]

            plt.plot(cumulative_generation_times, average_fitness_values, label=algorithm)

            # time_taken = all_results["Total time taken"]
            # num_explored_languages = all_results["Explored languages"]
            #
            # plt.scatter(time_taken, num_explored_languages, label=algorithm)

        plt.xlabel("Total time passed")
        plt.ylabel("Average generation fitness")
        plt.title("Effect of search algorithm on time and fitness")
        plt.legend(loc="upper left")
        plt.savefig("metasynthesis/programming_language/results/search_algorithm_comparison.jpg")
        plt.close()

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

            cumulative_generation_times = [sum(generation_times[0:i[0]]) for i in enumerate(generation_times)]

            plt.plot(cumulative_generation_times, average_fitness_values, label=setting)

        plt.xlabel("Total time passed")
        plt.ylabel("Average generation fitness")
        plt.title("Effect of search setting on time and fitness")
        plt.legend(loc="upper left")
        plt.savefig("metasynthesis/programming_language/results/search_setting_comparison.jpg")
        plt.close()

    def plot_search_timeout_performance(self):
        timeouts = [0.1, 0.5, 1, 2, 6, 10]

        general_genetic = EvolvingLanguage(max_search_time=self.best_parameters["max_search_time"],
                                           generation_size=self.best_parameters["generation_size"])
        start_population = general_genetic.generate_population()

        for timeout in timeouts:
            genetic = EvolvingLanguage(generation_limit=self.best_parameters["generation_limit"],
                                       generation_size=self.best_parameters["generation_size"],
                                       search_mode=self.best_parameters["search_mode"],
                                       max_search_time=timeout,
                                       crossover_probability=self.best_parameters["crossover_probability"],
                                       mutation_probability=self.best_parameters["mutation_probability"],
                                       dsl=self.dsl,
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
        plt.title("Effect of timeout on ratio correct")
        plt.legend(loc="upper left")
        plt.savefig("metasynthesis/programming_language/results/search_timeout_comparison_correct.jpg")

        plt.figure(2)
        plt.xlabel("Generation")
        plt.ylabel("Generation time taken")
        plt.title("Effect of timeout on generation time")
        plt.legend(loc="upper left")
        plt.savefig("metasynthesis/programming_language/results/search_timeout_comparison_time.jpg")

        plt.close('all')

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
        plt.close()

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
        plt.close()

    def plot_population_size_performance(self):
        generation_sizes = [10, 20, 30, 40, 50, 60]

        # Other important parameters
        generation_limit = 30

        for size in generation_sizes:
            genetic = EvolvingLanguage(generation_limit=generation_limit,
                                       generation_size=size,
                                       search_mode=self.best_parameters["search_mode"],
                                       max_search_time=self.best_parameters["max_search_time"],
                                       crossover_probability=self.best_parameters["crossover_probability"],
                                       mutation_probability=self.best_parameters["mutation_probability"],
                                       dsl=self.dsl)
            all_results = genetic.run_evolution()

            num_explored_languages = all_results["Explored languages"]
            generations, generation_times, best_fitness_values = extract_generation_time_and_fitness(all_results, "Best fitness")

            cumulative_generation_times = [sum(generation_times[0:i[0]]) for i in enumerate(generation_times)]

            plt.plot(cumulative_generation_times, best_fitness_values, label=str(size))

        plt.xlabel("Total time passed")
        plt.ylabel("Best fitness value")
        plt.title("Effect of population size on fitness vs time passed")
        plt.legend(loc="upper left")
        plt.savefig("metasynthesis/programming_language/results/population_size_comparison.jpg")
        plt.close()

    def plot_generation_limit_performance(self):
        generations = [5, 10]  # 20, 30, 40, 60

        for gen in generations:
            genetic = EvolvingLanguage(generation_limit=gen,
                                       generation_size=self.best_parameters["generation_size"],
                                       search_mode=self.best_parameters["search_mode"],
                                       max_search_time=self.best_parameters["max_search_time"],
                                       crossover_probability=self.best_parameters["crossover_probability"],
                                       mutation_probability=self.best_parameters["mutation_probability"],
                                       dsl=self.dsl)
            all_results = genetic.run_evolution()

            num_explored_languages = all_results["Explored languages"]
            generations, generation_times, best_fitness_values = extract_generation_time_and_fitness(all_results, "Best fitness")

            cumulative_generation_times = [sum(generation_times[0:i[0]]) for i in enumerate(generation_times)]

            plt.plot(cumulative_generation_times, best_fitness_values, label=str(gen))

        plt.xlabel("Total time passed")
        plt.ylabel("Best fitness value")
        plt.title("Effect of generation limit on fitness vs time passed")
        plt.legend(loc="upper left")
        plt.savefig("metasynthesis/programming_language/results/generation_limit_comparison.jpg")
        plt.close()

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
