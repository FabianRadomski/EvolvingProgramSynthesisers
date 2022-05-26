from typing import Callable

from common.environment.environment import Environment
from common.settings.robot_optimized_steps import RobotOptimizedSteps


class ObjectiveFun:
    def __init__(self, domain_name: str):
        self.domain_name = domain_name
        self.fun: Callable[[Environment, Environment], float]
        if domain_name in ['robot', 'pixel', 'string']:
            self.fun = self.env_dist_fun
        else:
            raise NotImplementedError("Unknown domain!")

    @staticmethod
    def env_dist_fun(env1: Environment, env2: Environment) -> float:
        return RobotOptimizedSteps().distance(env1, env2)

if __name__ == '__main__':
    print(ObjectiveFun('robot').fun)
