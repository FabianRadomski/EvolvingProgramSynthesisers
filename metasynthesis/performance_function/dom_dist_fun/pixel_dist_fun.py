from typing import Callable, List, Tuple

from common.environment.pixel_environment import PixelEnvironment
from metasynthesis.performance_function.dom_dist_fun.domain_dist_fun import DomainDistFun


def f1(x: PixelEnvironment, y: PixelEnvironment) -> float:
    # return number of coordinates for which x_i = 0 and y_i = 1
    assert len(x.pixels) == len(y.pixels)
    return sum([(e1 == 0 and e2 == 1) for (e1, e2) in zip(x.pixels, y.pixels)])


def f2(x: PixelEnvironment, y: PixelEnvironment) -> float:
    # return number of coordinates for which x_i = 0 and y_i = 0
    assert len(x.pixels) == len(y.pixels)
    return sum([(e1 == 0 and e2 == 0) for (e1, e2) in zip(x.pixels, y.pixels)])


def f3(x: PixelEnvironment, y: PixelEnvironment) -> float:
    # return number of coordinates for which x_i = 1 and y_i = 1
    assert len(x.pixels) == len(y.pixels)
    return sum([(e1 == 1 and e2 == 1) for (e1, e2) in zip(x.pixels, y.pixels)])


def f4(x: PixelEnvironment, y: PixelEnvironment) -> float:
    # return number of coordinates for which x_i = 1 and y_i = 0
    assert len(x.pixels) == len(y.pixels)
    return sum([(e1 == 1 and e2 == 0) for (e1, e2) in zip(x.pixels, y.pixels)])


def l1(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    # l1 distance
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def f5(x: PixelEnvironment, y: PixelEnvironment) -> float:
    # return manhattan distance between the pointers
    assert len(x.pixels) == len(y.pixels)
    return l1((x.x, x.y), (y.x, y.y))


class PixelDistFun(DomainDistFun):

    @staticmethod
    def fun_to_string() -> dict:
        return {f1: '01', f2: '00', f3: '11', f4: '10', f5: 'l1(p, p*)'}

    @staticmethod
    def partial_dist_funs() -> List[Callable[[PixelEnvironment, PixelEnvironment], float]]:
        return [f1, f2, f3, f4, f5]
