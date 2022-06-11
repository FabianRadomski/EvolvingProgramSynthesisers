from metasynthesis.programming_language.evolve_language import EvolvingLanguage

from common.program_synthesis.dsl import StandardDomainSpecificLanguage
from metasynthesis.programming_language.plot_statistics import LanguageStatistics


def get_search_setting(domain_for_func):
    search_setting = ""
    if domain_for_func == "pixel":
        search_setting = "PG"
    elif domain_for_func == "robot":
        search_setting = "RO"
    elif domain_for_func == "string":
        search_setting = "SO"

    return search_setting


def run_genetic_algorithm_once(domain_for_func):
    dsl = StandardDomainSpecificLanguage(domain)

    genetic = EvolvingLanguage(fitness_limit=1,
                               generation_limit=10,
                               crossover_probability=0.8,
                               mutation_probability=0.3,
                               elite_genomes=2,
                               generation_size=10,  # 10, 34
                               dsl=dsl,
                               search_setting=get_search_setting(domain_for_func),
                               max_search_time=0.01,
                               search_mode="debug",  # debug, param_train
                               search_algo="Brute",
                               print_stats=True)
    genetic.run_evolution()


if __name__ == '__main__':
    domain = "string"

    # run_genetic_algorithm_once(domain)

    stats = LanguageStatistics(domain=domain, print_stats=True, search_mode="param")

    stats.plot_search_algorithm_performance()
    stats.plot_search_setting_performance()
    # stats.plot_search_timeout_performance()
    # stats.plot_mutation_method_performance()
    # stats.plot_crossover_method_performance()
    # stats.plot_population_size_performance()
    # stats.plot_generation_limit_performance()
