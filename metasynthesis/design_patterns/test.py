from metasynthesis.design_patterns.evolving_design_patterns import EvolvingDesignPatterns

from common.program_synthesis.dsl import StandardDomainSpecificLanguage

domain = "robot"
x = "Brute"
y = "RG"
# x = "AS"

z = "small"

if __name__ == '__main__':
    genetic = EvolvingDesignPatterns(fitness_limit=1,
                                     generation_limit=30,
                                     crossover_probability=0.75,
                                     mutation_probability=0.004,
                                     generation_size=10,
                                     dsl=StandardDomainSpecificLanguage(domain),
                                     search_algo=x,
                                     search_setting=y,
                                     search_mode=z,
                                     max_search_time=1
                                     )

    genetic.run_evolution()
