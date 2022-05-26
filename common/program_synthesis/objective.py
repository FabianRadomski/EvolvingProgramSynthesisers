from typing import Callable

from common.environment import Environment


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
        return env1.distance(env2)
