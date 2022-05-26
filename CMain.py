from common.environment import RobotEnvironment
from common.program_synthesis.dsl import StandardDomainSpecificLanguage
from metasynthesis.language_constraints.constraints.ConstraintFactory import ConstraintFactory
from metasynthesis.language_constraints.genetic_algorithm.genetic_algorithm import ConstraintGeneticAlgorithm
from metasynthesis.language_constraints.properties.BinaryProperties import Independent, Identity
from metasynthesis.language_constraints.properties.PropertyFactory import PropertyFactory


def main():
    dsl = StandardDomainSpecificLanguage("robot")
    env1 = RobotEnvironment(5, 0, 0, 0, 0, False)
    env2 = RobotEnvironment(5, 0, 0, 0, 0, True)
    env3 = RobotEnvironment(5, 1, 0, 1, 1, False)
    env4 = RobotEnvironment(5, 2, 2, 1, 1, False)
    tests = [env1, env2, env3, env4]
    property_types = [Identity, Independent]
    factory = ConstraintFactory(PropertyFactory(dsl, property_types, tests))
    gen = ConstraintGeneticAlgorithm(1, 40, 0.05, factory.create(), 100)
    print(gen.run_evolution())

if __name__ == '__main__':
    main()