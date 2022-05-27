
from common.program_synthesis.dsl import StandardDomainSpecificLanguage
from typing import Callable

from common.environment.environment import Environment
from common.settings.settings import Settings
from common.environment.robot_environment import RobotEnvironment


class RobotOptimizedSteps(Settings):
    """Optimized amount of steps for robot environment."""

    def __init__(self):
        super().__init__("robot", StandardDomainSpecificLanguage('robot'))

    def distance(self, inp: RobotEnvironment, out: RobotEnvironment) -> float:
        if self.dist_fun is None:
            assert inp.size == out.size

            def d(xy1: 'tuple[int, int]', xy2: 'tuple[int, int]'):
                return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

            # position robot and position ball
            pr = (inp.rx, inp.ry)
            pb = (inp.bx, inp.by)

            # position goal robot and position goal bal
            pgr = (out.rx, out.ry)
            pgb = (out.bx, out.by)

            if pr != pb and pb != pgb:
                return d(pr, pb) + d(pb, pgb) + d(pgb, pgr) + 2
            elif pr == pb and pb != pgb:
                return d(pr, pgb) + d(pgb, pgr) + 1
            else:
                return d(pr, pgr)
        else:
            return self.dist_fun(inp, out)
