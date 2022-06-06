from typing import List, Dict

import matplotlib.pyplot as plt

from common.program_synthesis.dsl import StandardDomainSpecificLanguage
from metasynthesis.programming_language.evolve_language import EvolvingLanguage


class LanguageStatistics:

    def __init__(self, domain: str):
        self.domain = domain
        self.dsl = StandardDomainSpecificLanguage(domain)

    def plot_mutation_method_performance(self):
        # Add_construct, Add_random, Remove_random,
        mutation_params = [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.25, 0.25, 0.5], [0.4, 0.1, 0.5]]

        general_genetic = EvolvingLanguage()

        # Other important parameters
        cross_prob = 0
        mutation_prob = 0.3
        start_population = general_genetic.generate_population()

        for weights in mutation_params:

            genetic = EvolvingLanguage(crossover_probability=cross_prob,
                                       mutation_probability=mutation_prob,
                                       dsl=self.dsl,
                                       mutation_weights=weights,
                                       start_population=start_population)
            all_results = genetic.run_evolution()

            num_explored_languages = all_results["Explored languages"]
            generations, average_fitness_values = extract_generations_and_fitness(all_results)
            plt.plot(generations, average_fitness_values, label=str(weights))

        plt.xlabel("Generations")
        plt.ylabel("Average fitness")
        plt.title("Effect of mutation methods on fitness")
        plt.legend(loc="upper left")
        plt.style.use("ggplot")
        plt.show()


def extract_generations_and_fitness(all_results: Dict):
    generation_statistics = all_results["Generation statistics"]

    generations = []
    average_fitness_values = []
    for gen in generation_statistics.keys():
        generations.append(float(gen))
        average_fitness_values.append(generation_statistics[gen]["Average fitness"])

    return generations, average_fitness_values

def simple_plot_with_params(xs: List, ys: List, xlabel: str, ylabel: str, title: str):
    plt.plot(xs, ys)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
