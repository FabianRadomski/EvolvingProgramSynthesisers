from common.environment.robot_environment import RobotEnvironment
from common.program_synthesis.dsl import StandardDomainSpecificLanguage
from common.settings.settings import Settings

class RobotEntailment(Settings):

    def __init__(self):
        super().__init__("robot", StandardDomainSpecificLanguage('robot'))

    def distance(self, inp: RobotEnvironment, out: RobotEnvironment) -> float:
        return 0 if inp.rx == out.rx and \
                    inp.ry == out.ry and \
                    inp.bx == out.bx and \
                    inp.by == out.by and \
                    inp.holding == out.holding else 1