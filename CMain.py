import sys

from common.environment.pixel_environment import PixelEnvironment
from common.environment.robot_environment import RobotEnvironment
from common.environment.string_environment import StringEnvironment
from common.program_synthesis.dsl import StandardDomainSpecificLanguage
from metasynthesis.language_constraints.constraints.ConstraintFactory import ConstraintFactory
from metasynthesis.language_constraints.genetic_algorithm.genetic_algorithm import ConstraintGeneticAlgorithm
from metasynthesis.language_constraints.properties.BinaryProperties import Independent, Identity
from metasynthesis.language_constraints.properties.PropertyFactory import PropertyFactory
from solver.runner.algorithms import dicts
from solver.runner.runner import Runner


def get_test_domain(setting: chr):
    if setting == 'R':
        env1 = RobotEnvironment(5, 0, 0, 0, 0, False)
        env2 = RobotEnvironment(5, 0, 0, 0, 0, True)
        env3 = RobotEnvironment(5, 1, 0, 1, 1, False)
        env4 = RobotEnvironment(5, 2, 2, 1, 1, False)
        return [env1, env2, env3, env4]
    if setting == 'S':
        env1 = StringEnvironment(['S', 'T', 'R', 'I', 'N', 'G'], pos=3)
        env2 = StringEnvironment(['S', 'T', 'R', 'i', 'N', 'G'], pos=3)
        env3 = StringEnvironment(['s', 't', 'r', 'i', 'n', 'g'], pos=3)
        return [env1, env2, env3]
    if setting == 'P':
        env1 = PixelEnvironment(3, 3, 1, 1)
        env2 = PixelEnvironment(3, 3, 1, 1, [False, False, False, False, True, False, False, False, False])
        env3 = PixelEnvironment(3, 3, 0, 0)
        env4 = PixelEnvironment(3, 3, 0, 0, [True, True, True, True, False, True, True, True, True])
        return [env1, env2, env3, env4]


def get_dsl(setting: chr):
    if setting == 'R':
        return StandardDomainSpecificLanguage('robot')
    if setting == 'S':
        return StandardDomainSpecificLanguage('string')
    if setting == 'P':
        return StandardDomainSpecificLanguage('pixel')


def create_constraints(settings):
    tests = get_test_domain(settings['setting'][0])
    dsl = get_dsl(settings['setting'][0])
    factory = ConstraintFactory(PropertyFactory(dsl, [Identity, Independent], tests))
    return factory.create()


def create_runner(settings, dsl):
    runner = Runner(dicts(),
                    settings['algorithm'],
                    settings['setting'],
                    settings['test_cases'],
                    settings['time_limit'],
                    settings['debug'],
                    settings['store'],
                    dsl=dsl)
    return runner


def main():
    jobs = {}
    i = 1
    for alg in [["AS", "R", "E"],
                ["AS", "R", "O"],
                ["Brute", "P", "O"],
                ["Brute", "R", "E"],
                ["Brute", "R", "O"]]:
            jobs[i] = alg
            i+=1
    _, job_n = sys.argv
    algorithm, setting, objective_function = jobs[int(job_n)]
    print(sys.argv)
    run_constraint_verification(setting,
                                algorithm,
                                objective_function)


def run_constraint_verification(domain,
                                search_algorithm,
                                objective_function,
                                generations=40,
                                mutation_chance=0.1,
                                pop_size=20
                                ):
    if domain == 'P':
        domain_name = 'pixel'
    elif domain == 'S':
        domain_name = 'string'
    else:
        domain_name = 'robot'

    settings = {
        'algorithm': search_algorithm,
        'setting': domain + objective_function,
        'test_cases': 'param',
        'time_limit': 0.1,
        'debug': False,
        'store': True,
        'domain': domain_name
    }

    gen = ConstraintGeneticAlgorithm(1, generations, mutation_chance, create_constraints(settings), settings, pop_size)
    dsl = gen.run_evolution()
    logger = gen.logger

    for time in [0.1, 0.5, 1, 10]:
        data = {}
        settings = {
            'algorithm': search_algorithm,
            'setting': domain + objective_function,
            'test_cases': 'eval',
            'time_limit': time,
            'debug': False,
            'store': True,
            'domain': domain_name
        }
        runner = create_runner(settings, dsl)
        runner.run()
        data['constraints'] = runner.file_manager.written_data
        runner = create_runner(settings, get_dsl(domain))
        runner.run()
        data['normal'] = runner.file_manager.written_data

        logger.write_final(gen.best_chromosome[0], data, settings)


if __name__ == '__main__':
    main()
