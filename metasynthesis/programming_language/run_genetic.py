from metasynthesis.programming_language.evolving_language import EvolvingLanguage

from common.program_synthesis.dsl import StandardDomainSpecificLanguage

dsl = StandardDomainSpecificLanguage("robot")

if __name__ == '__main__':

    genetic = EvolvingLanguage(fitness_limit=1,
                               generation_limit=10,
                               crossover_probability=0.5,
                               mutation_probability=0.5,
                               generation_size=30,
                               dsl=dsl)
    genetic.run_evolution()

