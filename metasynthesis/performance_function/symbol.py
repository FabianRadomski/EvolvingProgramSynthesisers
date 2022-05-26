from typing import Callable

from operator import add, sub, mul, truediv as div

from common.environment.environment import Environment
from metasynthesis.performance_function.dom_dist_fun.pixel_dist_fun import PixelDistFun
from metasynthesis.performance_function.dom_dist_fun.robot_dist_fun import RobotDistFun
from metasynthesis.performance_function.dom_dist_fun.string_dist_fun import StringDistFun

Term = Callable[[Environment, Environment], float]
Operator = Callable[[float, float], float]


class Symbol:

    def __init__(self, symbol):
        self.symbol = symbol

    def is_operator(self) -> bool:
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError

    def __eq__(self, other):
        raise NotImplementedError


class OpSym(Symbol):

    def __init__(self, symbol: Operator):
        super().__init__(symbol)

    def is_operator(self):
        return True

    def __str__(self):
        return {add: '+', sub: '-', mul: '*', div: '/'}[self.symbol]

    def __eq__(self, other):
        if isinstance(other, OpSym):
            return self.symbol == other.symbol
        return False


class TermSym(Symbol):

    def __init__(self, symbol: Term):
        super().__init__(symbol)

    def is_operator(self):
        return False

    def __str__(self):
        domain = str(list(self.symbol.__annotations__.values())[0]).split('.')[-1].split('Env')[0]
        if domain == 'Robot':
            return RobotDistFun.fun_to_string()[self.symbol]
        elif domain == 'Pixel':
            return PixelDistFun.fun_to_string()[self.symbol]
        elif domain == 'String':
            return StringDistFun.fun_to_string()[self.symbol]
        else:
            raise Exception()

    def __eq__(self, other):
        if isinstance(other, TermSym):
            return self.symbol == other.symbol
        return False


# if __name__ == '__main__':
#     f1 = RobotDistFun.partial_dist_funs()[0]
#     f2 = PixelDistFun.partial_dist_funs()[0]
#     f3 = StringDistFun.partial_dist_funs()[0]
#     sym = TermSym(symbol=f1)
#     sym2 = TermSym(f2)
#     sym3 = TermSym(f3)
#     print(sym3)
#     print(sym2)
#     print(sym)
