import random
import time

import numpy as np

from common.environment import RobotEnvironment
from metasynthesis.performance_function.evolving_function import rand_env, EvolvingFunction
from metasynthesis.performance_function.expr_tree import ExpressionTree, generate_complete_tree, generate_high_tree, \
    printN
from metasynthesis.performance_function.dom_dist_fun.robot_dist_fun import RobotDistFun
from metasynthesis.performance_function.symbol import OpSym, TermSym
from operator import add, sub, mul, truediv as div


def test_high_vs_complete_tree():
    count = 0
    ct_1 = 0
    ct_2 = 0
    operators = [OpSym(add), OpSym(sub), OpSym(mul), OpSym(div)]
    for i in range(10000):
        env1 = rand_env()
        env2 = rand_env()
        t1 = time.time()
        d1 = generate_complete_tree(terms=terms, operators=operators, cur_depth=0, max_depth=8).distance_fun
        ct_1 += time.time() - t1
        t2 = time.time()
        d2 = generate_high_tree(terms=terms, operators=operators, cur_depth=0, max_depth=1000).distance_fun
        ct_2 += time.time() - t2
        if d1 == d2:
            count += 1
    print(count == 10000)
    print(ct_1, ct_2)


def test_random_walk():
    functions = RobotDistFun.partial_dist_funs()
    terms = list(map(lambda x: TermSym(x), functions))
    operators = [OpSym(add), OpSym(sub), OpSym(mul), OpSym(div)]
    a = []
    height = 4
    for i in range(10000):
        rand_expr = generate_complete_tree(terms=terms, operators=operators, cur_depth=0, max_depth=height)
        a.append(len(rand_expr.random_walk_uniform_depth(min_depth=2)[0]))
    a = np.array(a)
    print('Average depth:', np.mean(a))
    print([np.sum([a == i]) for i in range(0, height + 1)])


def height_of_random_trees():
    functions = RobotDistFun.partial_dist_funs()
    terms = list(map(lambda x: TermSym(x), functions))
    operators = [OpSym(add), OpSym(sub), OpSym(mul), OpSym(div)]
    a = []
    height = 6
    for i in range(10000):
        rand_expr = generate_complete_tree(terms=terms, operators=operators, cur_depth=0, max_depth=height)
        a.append(len(rand_expr.random_walk_uniform_depth(min_depth=2)[0]))
    a = np.array(a)
    print('Average', np.mean(a))
    print([np.sum([a == i]) for i in range(0, height + 1)])


def average_height_random_trees(num_pairs, max_depth=4):
    functions = RobotDistFun.partial_dist_funs()
    terms = list(map(lambda x: TermSym(x), functions))

    start_time = time.time()
    h = []
    h_children = []
    h_subtree = []
    for i in range(num_pairs):
        rand_expr1 = ExpressionTree.generate_random_expression(terms=terms, max_depth=max_depth)
        rand_expr2 = ExpressionTree.generate_random_expression(terms=terms, max_depth=max_depth)
        h.append(rand_expr2.height())
        h.append(rand_expr1.height())
        c1, c2 = rand_expr1.reproduce_with(rand_expr2)
        h_children.append(c1.height())
        h_children.append(c2.height())

        h_subtree.append(rand_expr1.random_walk_uniform_depth()[1].height())
        h_subtree.append(rand_expr2.random_walk_uniform_depth()[1].height())


    print('Average height of', num_pairs * 2, 'random trees:', np.mean(np.array(h)))
    print('Average height of random subtrees of ', num_pairs * 2, 'random trees:', np.mean(np.array(h_subtree)))
    height_frequency = [np.sum([np.array(h) == i]) for i in range(1, max_depth + 1)]
    print(height_frequency / np.sum(height_frequency))
    # print(np.sum(height_frequency))
    print('Average height of the resulting', num_pairs * 2, 'children:', np.mean(np.array(h_children)))
    print("--- %s seconds ---" % (time.time() - start_time))


def heights_of_random_trees_and_subtrees(max_depth=5, cardinality=2000):
    functions = RobotDistFun.partial_dist_funs()
    terms = list(map(lambda x: TermSym(x), functions))

    random_trees = [ExpressionTree.generate_random_expression(terms=terms, max_depth=max_depth) for _ in range(cardinality)]
    children = []
    random_subtrees_height = []
    random_subtrees_heights_at_depth_2 = []
    for parents in [random.sample(random_trees, 2) for _ in range(int(cardinality / 2))]:
        traversal_t, t = parents[0].random_walk_uniform_depth()
        if len(traversal_t) == 2:
            random_subtrees_heights_at_depth_2.append(t.height())
        traversal_t_prime, random_subtree = parents[1].random_walk_uniform_depth(min_depth=0)
        if len(traversal_t_prime) == 2:
            random_subtrees_heights_at_depth_2.append(random_subtree.height())
        random_subtrees_height.append(t.height())
        random_subtrees_height.append(random_subtree.height())
        c1 = parents[0].replace_branch(traversal=traversal_t, replace_with=random_subtree)
        c2 = parents[1].replace_branch(traversal=traversal_t_prime, replace_with=t)
        children.append(c1)
        children.append(c2)

    print('Average height of', cardinality, 'random trees:', np.mean(np.array([x.height() for x in random_trees])))
    print('Max height of', cardinality, 'random trees:', np.max(np.array([x.height() for x in random_trees])))
    print('Min height of', cardinality, 'random trees:', np.min(np.array([x.height() for x in random_trees])))
    print('-------------')
    print('Average height of random subtrees:', np.mean(np.array(random_subtrees_height)))

    height_frequency = [np.sum([np.array(random_subtrees_height) == i]) for i in range(0, max_depth + 1)]
    print(height_frequency / np.sum(height_frequency))
    print(np.sum(height_frequency / np.sum(height_frequency)))
    print('heights at depth 2:')
    temp = [np.sum([np.array(random_subtrees_heights_at_depth_2) == i]) for i in range(0, max_depth + 1)]
    print(temp / np.sum(temp))
    print(np.sum(temp / np.sum(temp)))

    print('Max height of random subtrees:', np.max(np.array(random_subtrees_height)))
    print('Min height of random subtrees:', np.min(np.array(random_subtrees_height)))
    print('-------------')
    print('Average height of children:', np.mean(np.array([x.height() for x in children])))
    print('Max height of children:', np.max(np.array([x.height() for x in children])))
    print('Min height of children:', np.min(np.array([x.height() for x in children])))

    # children_prime = []
    # random_subtrees_height_prime = []
    # for parents in [random.sample(children, 2) for _ in range(int(cardinality / 2))]:
    #     traversal_t, t = parents[0].random_walk_uniform_depth()
    #     traversal_t_prime, random_subtree = parents[1].random_walk_uniform_depth(min_depth=0)
    #     random_subtrees_height_prime.append(t.height())
    #     random_subtrees_height_prime.append(random_subtree.height())
    #     c1 = parents[0].replace_branch(traversal=traversal_t, replace_with=random_subtree)
    #     c2 = parents[1].replace_branch(traversal=traversal_t_prime, replace_with=t)
    #     children_prime.append(c1)
    #     children_prime.append(c2)
    #
    # print('Average height of random subtrees:', np.mean(np.array(random_subtrees_height_prime)))
    # height_frequency = [np.sum([np.array(random_subtrees_height_prime) == i]) for i in range(0, max_depth + 1)]
    # print(height_frequency / np.sum(height_frequency))
    # print('Max height of random subtrees:', np.max(np.array(random_subtrees_height_prime)))
    # print('Min height of random subtrees:', np.min(np.array(random_subtrees_height_prime)))
    # print('-------------')
    # print('Average height of children:', np.mean(np.array([x.height() for x in children_prime])))
    # print('Max height of children:', np.max(np.array([x.height() for x in children_prime])))
    # print('Min height of children:', np.min(np.array([x.height() for x in children_prime])))
    #
    #
    # children_prime_prime = []
    # random_subtrees_height_prime_prime = []
    # for parents in [random.sample(children_prime, 2) for _ in range(int(cardinality / 2))]:
    #     traversal_t, t = parents[0].random_walk_uniform_depth()
    #     traversal_t_prime, random_subtree = parents[1].random_walk_uniform_depth(min_depth=0)
    #     random_subtrees_height_prime_prime.append(t.height())
    #     random_subtrees_height_prime_prime.append(random_subtree.height())
    #     c1 = parents[0].replace_branch(traversal=traversal_t, replace_with=random_subtree)
    #     c2 = parents[1].replace_branch(traversal=traversal_t_prime, replace_with=t)
    #     children_prime_prime.append(c1)
    #     children_prime_prime.append(c2)
    #
    # print('Average height of random subtrees:', np.mean(np.array(random_subtrees_height_prime_prime)))
    # height_frequency = [np.sum([np.array(random_subtrees_height_prime_prime) == i]) for i in range(0, max_depth + 1)]
    # print(height_frequency / np.sum(height_frequency))
    # print('Max height of random subtrees:', np.max(np.array(random_subtrees_height_prime_prime)))
    # print('Min height of random subtrees:', np.min(np.array(random_subtrees_height_prime_prime)))
    # print('-------------')
    # print('Average height of children:', np.mean(np.array([x.height() for x in children_prime_prime])))
    # print('Max height of children:', np.max(np.array([x.height() for x in children_prime_prime])))
    # print('Min height of children:', np.min(np.array([x.height() for x in children_prime_prime])))


if __name__ == '__main__':
    # weights = list(uniform(0, 1, len(partial_dist_functions)).astype(float))
    functions = RobotDistFun.partial_dist_funs()
    terms = list(map(lambda x: TermSym(x), functions))
    operators = [OpSym(add), OpSym(sub), OpSym(mul), OpSym(div)]

    # average_height_random_trees(num_pairs=1000, max_depth=5)
    # TODO add weights to terms (weight field of TermSym)
    # TODO do not replace with the same term
    # TODO what to do with negative numbers (for now I return the absolute value)

    # print(generate_complete_tree(terms=terms, operators=operators, cur_depth=0, max_depth=3))
    # print(generate_high_tree(terms=terms, operators=operators, cur_depth=0, max_depth=7))
    # test_high_vs_complete_tree()
    #
    # ga = EvolvingFunction(fitness_limit=0, generation_limit=100, crossover_probability=0.7,
    #                       mutation_probability=0.01, generation_size=50, domain_name='robot',
    #                       tournament_size=3, elite_size=8)
    # print(ga.run_evolution())
    # test_random_walk()
    # test_tall_vs_wide_tree()
    heights_of_random_trees_and_subtrees(max_depth=10, cardinality=20000)



