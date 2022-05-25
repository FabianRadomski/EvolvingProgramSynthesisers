from typing import Callable, List

from common.environment import StringEnvironment
from metasynthesis.performance_function.dom_dist_fun.domain_dist_fun import DomainDistFun


def f1(x: StringEnvironment, y: StringEnvironment) -> float:
    return 0


class StringDistFun(DomainDistFun):

    @staticmethod
    def fun_to_string() -> dict:
        return {f1: 'String function'}

    @staticmethod
    def partial_dist_funs() -> List[Callable[[StringEnvironment, StringEnvironment], float]]:
        return [f1]
