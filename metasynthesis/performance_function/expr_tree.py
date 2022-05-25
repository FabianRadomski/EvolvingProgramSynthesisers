import random
import sys
from operator import mul, add, sub, truediv as div
from random import choice
from typing import List, Tuple
import numpy as np

from common.environment import Environment
from metasynthesis.performance_function.dom_dist_fun.robot_dist_fun import RobotDistFun
from metasynthesis.performance_function.symbol import Symbol, OpSym, TermSym
import copy


def get_next_bernoulli(p=0.5):
    return random.uniform(0, 1) <= p


class ExpressionTree:
    def __init__(self, symbol: Symbol, left=None, right=None):
        self.symbol = symbol
        self.left = left
        self.right = right

    def __str__(self) -> str:
        if self.symbol.is_operator():
            return '(' + str(self.left) + ' ' + str(self.symbol) + ' ' + str(self.right) + ')'
        else:  # it is a terminal symbol
            return str(self.symbol)

    def eval(self, x: Environment, y: Environment) -> float:
        try:
            fun = self.symbol.symbol
            if self.symbol.is_operator():
                return fun(self.left.eval(x, y), self.right.eval(x, y))
            else:  # it is a terminal symbol
                return fun(x, y)
        except ZeroDivisionError:
            return sys.float_info.max

    def distance_fun(self, x: Environment, y: Environment) -> float:
        return abs(self.eval(x=x, y=y))

    def random_walk_bernoulli_helper(self, cur_depth, min_depth, traversal) -> Tuple[List[int], 'ExpressionTree']:
        # returns a random subtree
        stop: bool = get_next_bernoulli(p=0.3)  # true with probability p
        if (stop and (cur_depth >= min_depth)) or (not self.symbol.is_operator()):
            # stop if (stop is true and (cur_depth >= min_depth)) or we reached a leaf
            return traversal, self
        else:
            pick_child, idx = (self.left, 0) if get_next_bernoulli(p=0.5) else (self.right, 1)
            traversal.append(idx)
            return pick_child.random_walk_bernoulli_helper(cur_depth=cur_depth + 1, min_depth=min_depth,
                                                           traversal=traversal)

    def random_walk_bernoulli(self, min_depth=0) -> Tuple[List[int], 'ExpressionTree']:
        return self.random_walk_bernoulli_helper(cur_depth=0, min_depth=min_depth, traversal=[])

    def random_walk_uniform_depth_helper(self, min_depth) -> Tuple[List[int], 'ExpressionTree']:
        # returns a random subtree
        length_of_traversal = random.randint(min_depth, self.height())
        if length_of_traversal == 0:
            return [], copy.deepcopy(self)
        rand_traversal = np.random.randint(0, 2, (length_of_traversal,))
        result = copy.deepcopy(self)
        for idx, dir in enumerate(list(rand_traversal)):
            if not result.symbol.is_operator():
                return rand_traversal[:idx], result
            else:
                if dir == 1:
                    result = result.right
                else:
                    result = result.left
        return rand_traversal, result

    def random_walk_uniform_depth(self, min_depth=0) -> Tuple[List[int], 'ExpressionTree']:
        return self.random_walk_uniform_depth_helper(min_depth=min_depth)

    def get_ops_helper(self, traversal: List[int]) -> List[Tuple[List[int], 'ExpressionTree']]:
        if self.symbol.is_operator():
            res: Tuple[List[int], ExpressionTree] = (traversal, self)
            return [res] + self.left.get_ops_helper(traversal=traversal + [0]) + self.right.get_ops_helper(
                traversal=traversal + [1])
        else:
            return []

    def get_operators(self) -> List[Tuple[List[int], 'ExpressionTree']]:
        return self.get_ops_helper(traversal=[])

    def get_terms_helper(self, traversal: List[int]) -> List[Tuple[List[int], 'ExpressionTree']]:
        if self.symbol.is_operator():
            return self.left.get_terms_helper(traversal=traversal + [0]) + self.right.get_terms_helper(
                traversal=traversal + [1])
        else:
            return [(traversal, self)]

    def get_terms(self) -> List[Tuple[List[int], 'ExpressionTree']]:
        return self.get_terms_helper(traversal=[])

    def mutate_tree(self, verbose=False) -> 'ExpressionTree':
        # possible mutations:
        # a) pick a random operator (internal node) and change it
        # b) pick a random term (leaf) and change it
        # c) TODO: pick a random term (leaf) and replace it with a random operator with two random terms as children
        clone = copy.deepcopy(self)
        traversal, random_node = clone.random_walk_uniform_depth(min_depth=0)
        if random_node.symbol.is_operator():
            if verbose:
                print('Changing:', random_node.symbol)
            operators = [add, sub, mul, div]
            operators.remove(random_node.symbol.symbol)
            random_node.symbol = OpSym(choice(operators))
            if verbose:
                print('New node:', random_node.symbol)
        else:
            if verbose:
                print('Changing:', random_node.symbol)
            funs = RobotDistFun.partial_dist_funs()
            funs.remove(random_node.symbol.symbol)
            random_node.symbol = TermSym(choice(funs))
            if verbose:
                print('New node:', random_node.symbol)
        return clone

    def reproduce_with_t_with_t_op_with_op(self, other: 'ExpressionTree', verbose=False) -> Tuple['ExpressionTree', 'ExpressionTree']:
        # terms are replaced only with terms, operators are replaced only with operators
        traversal_t, t = self.random_walk_uniform_depth()
        if verbose:
            print('Traversal in T:', traversal_t)
            print('Node to be replaced:', t)
        if t.symbol.is_operator():
            traversal_sym_nodes: List[Tuple[List[int], ExpressionTree]] = other.get_operators()
        else:
            traversal_sym_nodes: List[Tuple[List[int], ExpressionTree]] = other.get_terms()
        traversal_t_prime, random_symbol = choice(traversal_sym_nodes)
        if verbose:
            print('Traversal in T\'', traversal_t_prime)
            print('Random node in other:', random_symbol)
        c1 = self.replace_branch(traversal=traversal_t, replace_with=random_symbol)
        c2 = other.replace_branch(traversal=traversal_t_prime, replace_with=t)
        return c1, c2

    def reproduce_with(self, other: 'ExpressionTree', verbose=False) -> Tuple['ExpressionTree', 'ExpressionTree']:
        traversal_t, t = self.random_walk_uniform_depth()
        if verbose:
            print('Traversal in T:', traversal_t)
            print('Node to be replaced:', t)
        # term/op can be replace with term/op
        traversal_t_prime, random_subtree = other.random_walk_uniform_depth(min_depth=0)
        if verbose:
            print('Traversal in T\'', traversal_t_prime)
            print('Random node in other:', random_subtree)
        c1 = self.replace_branch(traversal=traversal_t, replace_with=random_subtree)
        c2 = other.replace_branch(traversal=traversal_t_prime, replace_with=t)
        return c1, c2

    def replace_branch(self, traversal: List[int], replace_with: 'ExpressionTree') -> 'ExpressionTree':
        if len(traversal) == 0:
            return replace_with
        elif len(traversal) == 1:
            if traversal[0] == 0:
                # replace the left child
                return ExpressionTree(self.symbol, replace_with, self.right)
            else:
                # replace the right child
                return ExpressionTree(self.symbol, self.left, replace_with)
        else:
            if traversal[0] == 0:
                return ExpressionTree(self.symbol,
                                      self.left.replace_branch(traversal=traversal[1:], replace_with=replace_with),
                                      self.right)
            else:
                return ExpressionTree(self.symbol, self.left,
                                      self.right.replace_branch(traversal=traversal[1:], replace_with=replace_with))

    def height(self):
        if self.symbol.is_operator():
            return max(self.left.height(), self.right.height()) + 1
        else:
            return 0

    @staticmethod
    def generate_random_expression(terms: List[TermSym],
                                   operators: List[OpSym] = [OpSym(add), OpSym(sub), OpSym(mul), OpSym(div)],
                                   max_depth=4) -> 'ExpressionTree':
        return helper_generator(terms=terms, operators=operators, cur_depth=0, max_depth=max_depth)

    def __eq__(self, other):
        if isinstance(other, ExpressionTree):
            return self.symbol.__eq__(other.symbol) and self.left.__eq__(other.left) and self.right.__eq__(other.right)
        return False


def printN(arr: List[ExpressionTree]):
    return list(map(lambda x: str(x), arr))


def helper_generator(terms, operators, cur_depth, max_depth) -> ExpressionTree:
    if cur_depth == max_depth:
        return ExpressionTree(symbol=choice(terms), left=None, right=None)
    else:
        # min depth = 1
        probability_of_operator = 1 if (cur_depth == 0) else 0.6  # TODO: tune this probability
        bernoulli = get_next_bernoulli(probability_of_operator)
        if bernoulli:
            symbol = choice(operators)
            return ExpressionTree(symbol=symbol,
                                  left=helper_generator(terms, operators, cur_depth + 1, max_depth),
                                  right=helper_generator(terms, operators, cur_depth + 1, max_depth))
        else:
            symbol = choice(terms)
            return ExpressionTree(symbol=symbol,
                                  left=None,
                                  right=None)


def generate_complete_tree(terms, operators, cur_depth, max_depth) -> ExpressionTree:
    if cur_depth == max_depth:
        return ExpressionTree(symbol=choice(terms), left=None, right=None)
    else:
        symbol = choice(operators)
        return ExpressionTree(symbol=symbol,
                              left=generate_complete_tree(terms, operators, cur_depth + 1, max_depth),
                              right=generate_complete_tree(terms, operators, cur_depth + 1, max_depth))


def generate_high_tree(terms, operators, cur_depth, max_depth) -> ExpressionTree:
    if cur_depth == max_depth:
        return ExpressionTree(symbol=choice(terms), left=None, right=None)
    else:
        symbol = choice(operators)
        return ExpressionTree(symbol=symbol,
                              left=generate_high_tree(terms, operators, cur_depth + 1, max_depth),
                              right=ExpressionTree(symbol=choice(terms), left=None, right=None))
