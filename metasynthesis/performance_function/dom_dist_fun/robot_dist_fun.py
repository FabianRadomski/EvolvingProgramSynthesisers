from typing import Tuple, Callable, List

from common.environment.robot_environment import RobotEnvironment
from metasynthesis.performance_function.dom_dist_fun.domain_dist_fun import DomainDistFun


def l1(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    # l1 distance
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def pr(x: RobotEnvironment):
    # robot position
    return x.rx, x.ry


def pb(x: RobotEnvironment):
    # ball position
    return x.bx, x.by


def f1(x: RobotEnvironment, y: RobotEnvironment) -> float:
    return l1(pr(x), pb(y))


def f2(x: RobotEnvironment, y: RobotEnvironment) -> float:
    return l1(pr(x), pr(y))


def f3(x: RobotEnvironment, y: RobotEnvironment) -> float:
    return l1(pb(y), pr(y))


def f4(x: RobotEnvironment, y: RobotEnvironment) -> float:
    return l1(pr(x), pb(x))


def f5(x: RobotEnvironment, y: RobotEnvironment) -> float:
    return l1(pb(x), pb(y))


class RobotDistFun(DomainDistFun):

    @staticmethod
    def fun_to_string() -> dict:
        return {f1: 'd(r, b*)', f2: 'd(r, r*)', f3: 'd(b*, r*)', f4: 'd(r, b)', f5: 'd(b, b*)'}

    @staticmethod
    def partial_dist_funs() -> List[Callable[[RobotEnvironment, RobotEnvironment], float]]:
        return [f1, f2, f3, f4, f5]
