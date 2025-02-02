from metasynthesis.programming_language.evolving_language import EvolvingLanguage

from common.program_synthesis.dsl import StandardDomainSpecificLanguage

dsl = StandardDomainSpecificLanguage("pixel")

if __name__ == '__main__':

    genetic = EvolvingLanguage(fitness_limit=1,
                               generation_limit=30,
                               crossover_probability=0.8,
                               mutation_probability=0.3,
                               elite_genomes=2,
                               generation_size=10,
                               dsl=dsl,
                               test_cases_per_genome=30,
                               max_search_time=1)
    genetic.run_evolution()

