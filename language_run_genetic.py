from metasynthesis.programming_language.evolve_language import EvolvingLanguage

from common.program_synthesis.dsl import StandardDomainSpecificLanguage
from metasynthesis.programming_language.plot_statistics import LanguageStatistics

domain = "string"

search_setting = ""
if domain == "pixel":
    search_setting = "PG"
elif domain == "robot":
    search_setting = "RO"
elif domain == "string":
    search_setting = "SO"

dsl = StandardDomainSpecificLanguage(domain)

if __name__ == '__main__':

    # genetic = EvolvingLanguage(fitness_limit=1,
    #                            generation_limit=5,
    #                            crossover_probability=0.8,
    #                            mutation_probability=0.3,
    #                            elite_genomes=2,
    #                            generation_size=6,  # 10, 34
    #                            dsl=dsl,
    #                            search_setting=search_setting,
    #                            max_search_time=0.1,
    #                            search_mode="debug",  # set to "eval" for final, "debug" for debugging
    #                            search_algo="Brute")
    # genetic.run_evolution()

    stats = LanguageStatistics(domain)

    # stats.plot_search_algorithm_performance()
    # stats.plot_search_setting_performance()
    # stats.plot_search_timeout_performance()
    # stats.plot_mutation_method_performance()
    # stats.plot_crossover_method_performance()
    # stats.plot_population_size_performance()
    stats.plot_generation_limit_performance()
