from metasynthesis.programming_language.evolving_language import EvolvingLanguage

from common.program_synthesis.dsl import StandardDomainSpecificLanguage

dsl = StandardDomainSpecificLanguage("robot")


genetic = EvolvingLanguage(1, 100, 0.5, 0.5, 2, dsl)

genetic.run_evolution()

