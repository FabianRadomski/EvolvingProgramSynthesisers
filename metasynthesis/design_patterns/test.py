from metasynthesis.design_patterns.evolving_design_patterns import EvolvingDesignPatterns

from common.program_synthesis.dsl import StandardDomainSpecificLanguage

dsl = StandardDomainSpecificLanguage("robot")

if __name__ == '__main__':

    genetic = EvolvingDesignPatterns(fitness_limit=1,
                               generation_limit=30,
                               crossover_probability=0.8,
                               mutation_probability=0.3,
                               generation_size=10,
                               dsl=dsl)
    # fd = genetic.generate_function(3, 1)
    # print(fd)
    g = genetic.generate_genome(4)
    for p in g:
        print(p)
    # population = genetic.generate_population()
    # print(population)
