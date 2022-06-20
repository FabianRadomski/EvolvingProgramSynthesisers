import numpy as np
import matplotlib.pyplot as plt

from metasynthesis.design_patterns.evolving_design_patterns import EvolvingDesignPatterns
from common.program_synthesis.dsl import StandardDomainSpecificLanguage


class EvaluationDP:
    def __init__(self, domain: str, test_cases: str):
        self.domain = domain
        self.test_cases = test_cases
        self.baseDSL = StandardDomainSpecificLanguage(domain)

        self.all_params = {"generation_limit": 30,
                           "generation_size": 50,
                           "elite_genomes": 5,
                           "test_cases": test_cases,
                           "search_algorithm": "AS",
                           "search_setting": "SG",
                           "time_limit_sec": 1,
                           "crossover_probability": 0.85,
                           "mutation_probability": 0.003,
                           }

    def get_settings_from_domain(self):
        start = self.domain[0].upper()
        end = ["G", "O", "E"]
        return [start+x for x in end]

    def plot_evolution_with_time_limit(self):
        timeouts = [0.1, 0.2, 0.5, 1]
        genetic = EvolvingDesignPatterns(generation_limit=self.all_params["generation_limit"],
                                         crossover_probability=self.all_params["crossover_probability"],
                                         mutation_probability=self.all_params["mutation_probability"],
                                         generation_size=self.all_params["generation_size"],
                                         elite_genomes=self.all_params["elite_genomes"],
                                         dsl=self.baseDSL,
                                         algo=self.all_params["search_algorithm"],
                                         setting=self.all_params["search_setting"],
                                         test_cases=self.all_params["test_cases"],
                                         time_limit_sec=self.all_params["time_limit_sec"]
                                         )
        for timeout in timeouts:

            res = genetic.run_evolution()
            print(res)

            # fig, ax = plt.subplots()
            x = np.arange(0, self.all_params["generation_limit"])
            y = res["avg_fitnesses"]
            plt.plot(x, y, label=timeout)


        plt.xlabel("Generation")
        plt.ylabel("Average fitness")
        plt.legend(loc="lower right")
        plt.show()


    def compare_settings(self):

        settings = self.get_settings_from_domain()

        for setting in settings:
            print(setting)
            max_fitness = []

            genetic = EvolvingDesignPatterns(generation_limit=self.all_params["generation_limit"],
                                             crossover_probability=self.all_params["crossover_probability"],
                                             mutation_probability=self.all_params["mutation_probability"],
                                             generation_size=self.all_params["generation_size"],
                                             elite_genomes=self.all_params["elite_genomes"],
                                             dsl=self.baseDSL,
                                             algo=self.all_params["search_algorithm"],
                                             setting=setting,
                                             test_cases=self.all_params["test_cases"],
                                             time_limit_sec=self.all_params["time_limit_sec"]
                                             )

            for i in range(5):
                print(i)
                res = genetic.run_evolution()
                max_fitness.append(max(res["max_fitnesses"]))
            print("average max fitness: ", np.mean(max_fitness))


    def compare_algorithms(self):
        algorithms = ["AS", "Brute", "MH", "LNS"]

        for algo in algorithms:
            genetic = EvolvingDesignPatterns(generation_limit=self.all_params["generation_limit"],
                                             crossover_probability=self.all_params["crossover_probability"],
                                             mutation_probability=self.all_params["mutation_probability"],
                                             generation_size=self.all_params["generation_size"],
                                             elite_genomes=self.all_params["elite_genomes"],
                                             dsl=self.baseDSL,
                                             algo=algo,
                                             setting=self.all_params["search_setting"],
                                             test_cases=self.all_params["test_cases"],
                                             time_limit_sec=self.all_params["time_limit_sec"]
                                             )
            res = genetic.run_evolution()


    def choose_mutation_probability(self):
        mp = [0.001, 0.003, 0.005, 0.010]

        for mutation_probability in mp:
            print(mutation_probability)
            max_fitness = []
            genetic = EvolvingDesignPatterns(generation_limit=self.all_params["generation_limit"],
                                             crossover_probability=self.all_params["crossover_probability"],
                                             mutation_probability=mutation_probability,
                                             generation_size=self.all_params["generation_size"],
                                             elite_genomes=self.all_params["elite_genomes"],
                                             dsl=self.baseDSL,
                                             algo=self.all_params["search_algorithm"],
                                             setting=self.all_params["search_setting"],
                                             test_cases=self.all_params["test_cases"],
                                             time_limit_sec=self.all_params["time_limit_sec"]
                                             )

            for i in range(5):
                print(i)
                res = genetic.run_evolution()
                max_fitness.append(max(res["max_fitnesses"]))
            print("average max fitness: ", np.mean(max_fitness))

    def choose_crossover_probability(self):
        cp = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

        for crossover_probability in cp:
            print(crossover_probability)
            max_fitness = []
            genetic = EvolvingDesignPatterns(generation_limit=self.all_params["generation_limit"],
                                             crossover_probability=crossover_probability,
                                             mutation_probability=self.all_params["mutation_probability"],
                                             generation_size=self.all_params["generation_size"],
                                             elite_genomes=self.all_params["elite_genomes"],
                                             dsl=self.baseDSL,
                                             algo=self.all_params["search_algorithm"],
                                             setting=self.all_params["search_setting"],
                                             test_cases=self.all_params["test_cases"],
                                             time_limit_sec=self.all_params["time_limit_sec"]
                                             )

            for i in range(5):
                print(i)
                res = genetic.run_evolution()
                max_fitness.append(max(res["max_fitnesses"]))
            print("average max fitness: ", np.mean(max_fitness))




if __name__ == '__main__':
    e = EvaluationDP("string", "DP_test")
    e.plot_evolution_with_time_limit()
    # e.run_single_evolution()