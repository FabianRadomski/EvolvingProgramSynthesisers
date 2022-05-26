from typing import List, Callable

from common.environment import Environment


class DomainDistFun(object):

    @staticmethod
    def partial_dist_funs() -> List[Callable[[Environment, Environment], float]]:
        raise NotImplementedError()

    @staticmethod
    def fun_to_string() -> dict:
        raise NotImplementedError()
