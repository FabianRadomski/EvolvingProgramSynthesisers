from typing import Callable, List

from common.environment import PixelEnvironment
from metasynthesis.performance_function.dom_dist_fun.domain_dist_fun import DomainDistFun


def f1(x: PixelEnvironment, y: PixelEnvironment) -> float:
    return 0


class PixelDistFun(DomainDistFun):

    @staticmethod
    def fun_to_string() -> dict:
        return {f1: 'Pixelfunction'}

    @staticmethod
    def partial_dist_funs() -> List[Callable[[PixelEnvironment, PixelEnvironment], float]]:
        return [f1]
