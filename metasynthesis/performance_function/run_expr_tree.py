import time

from common.environment import RobotEnvironment
from metasynthesis.performance_function.expr_tree import ExpressionTree
from metasynthesis.performance_function.dom_dist_fun.robot_dist_fun import RobotDistFun
from metasynthesis.performance_function.symbol import OpSym, TermSym
from operator import add, sub, mul, truediv as div

if __name__ == '__main__':
    # weights = list(uniform(0, 1, len(partial_dist_functions)).astype(float))
    functions = RobotDistFun.partial_dist_funs()
    terms = list(map(lambda x: TermSym(x), functions))

    start_time = time.time()
    for i in range(100):
        rand_expr1 = ExpressionTree.generate_random_expression(terms=terms, max_depth=3)
        rand_expr2 = ExpressionTree.generate_random_expression(terms=terms, max_depth=3)
        print('P1:', rand_expr1)
        print('P2:', rand_expr2)
        c1, c2 = rand_expr1.reproduce_with(rand_expr2, verbose=True)
        print('C1:', c1)
        print('C2:', c2)
        print('------------------')

    print("--- %s seconds ---" % (time.time() - start_time))

    root = ExpressionTree.generate_random_expression(terms=terms, max_depth=2)
    env1: RobotEnvironment = RobotEnvironment(5, 2, 1, 4, 3, 0)
    env2: RobotEnvironment = RobotEnvironment(5, 3, 1, 4, 5, 0)
    print('Expression:', root)
    print('Evaluate expression for the given Robot environments:', root.distance_fun(env1, env2))
    mutated = root.mutate_tree(verbose=True)
    print('Mutated expression:', mutated)


    a = ExpressionTree(symbol=OpSym(div), left=ExpressionTree(terms[0], None, None), right=ExpressionTree(terms[1], None, None))
    env1: RobotEnvironment = RobotEnvironment(5, 2, 1, 4, 3, 0)
    env2: RobotEnvironment = RobotEnvironment(5, 2, 1, 4, 5, 0)
    print('herhe', a.distance_fun(env1, env2))
    print(a)

    # TODO add weights to terms (weight field of TermSym)
    # TODO do not replace with the same term
    # TODO what to do with negative numbers (for now I return the absolute value)