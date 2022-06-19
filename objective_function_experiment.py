import time
from metasynthesis.performance_function.dom_dist_fun.string_dist_fun import StringDistFun
from metasynthesis.performance_function.dom_dist_fun.robot_dist_fun import RobotDistFun
from metasynthesis.performance_function.evolving_function import EvolvingFunction
from metasynthesis.performance_function.expr_tree import ExpressionTree
from metasynthesis.performance_function.symbol import TermSym
from datetime import datetime


def random_function_robot():
    functions = RobotDistFun.partial_dist_funs()
    terms = list(map(lambda x: TermSym(x), functions))
    return ExpressionTree.generate_random_expression(terms=terms, max_depth=3)


def random_function_string():
    functions = StringDistFun.partial_dist_funs()
    terms = list(map(lambda x: TermSym(x), functions))
    return ExpressionTree.generate_random_expression(terms=terms, max_depth=3)


def run_experiment(domain: str, num_generations, pop_size, p_c, p_m,
                   tournament_size, elite_size, w1, w2, w3, d_max, pr_op, time_out):
    date = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    outF = open('obj_fun_experiments/%s_%s' % (domain, date), "w")
    outF.write('domain=%s' % domain)
    outF.write("\n")
    outF.write('num_generations=%s' % num_generations)
    outF.write("\n")
    outF.write('pop_size=%s' % pop_size)
    outF.write("\n")
    outF.write('p_c=%s' % p_c)
    outF.write("\n")
    outF.write('p_m=%s' % p_m)
    outF.write("\n")
    outF.write('tournament_size=%s' % tournament_size)
    outF.write("\n")
    outF.write('elite_size=%s' % elite_size)
    outF.write("\n")
    outF.write('w1=%s' % w1)
    outF.write("\n")
    outF.write('w2=%s' % w2)
    outF.write("\n")
    outF.write('w3=%s' % w3)
    outF.write("\n")
    outF.write('d_max=%s' % d_max)
    outF.write("\n")
    outF.write('pr_op=%s' % pr_op)
    outF.write("\n")
    outF.write('time_out=%s' % time_out)
    outF.write("\n")
    start_time = time.time()
    ga = EvolvingFunction(fitness_limit=0, generation_limit=num_generations, crossover_probability=p_c,
                          mutation_probability=p_m, generation_size=pop_size,
                          domain=domain, tournament_size=tournament_size, elite_size=elite_size,
                          w1=w1, w2=w2, w3=w3, d_max=d_max, pr_op=pr_op, brute_time_out=time_out)
    avg_fit, best_fit, best_individual, evaluation_best, evaluation_manually_designed = ga.run_evolution()
    ga_execution_time_minutes = (time.time() - start_time) / 60.0
    outF.write('avg_fit=%s' % str(avg_fit))
    outF.write("\n")
    outF.write('best_fit=%s' % str(best_fit))
    outF.write("\n")
    outF.write('best_individual=%s' % str(best_individual))
    outF.write("\n")
    outF.write('evaluation_best=%s' % str(evaluation_best))
    outF.write("\n")
    outF.write('evaluation_manually_designed=%s' % str(evaluation_manually_designed))
    outF.write("\n")
    outF.write('ga_execution_time_minutes=%s' % str(ga_execution_time_minutes))
    outF.close()


if __name__ == '__main__':
    run_experiment(domain='robot', num_generations=1, pop_size=6, p_c=0.5,
                   p_m=0.08, tournament_size=3, elite_size=2, w1=0.9, w2=0,
                   w3=0.1, d_max=2, pr_op=0.7, time_out=1)
