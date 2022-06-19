from typing import Callable

from common.environment.environment import Environment
from common.program_synthesis.dsl import DomainSpecificLanguage
from common.tokens.abstract_tokens import Token, TransToken, BoolToken


class Settings:
    """Abstract settings class."""

    def __init__(self, domain: str, dsl: DomainSpecificLanguage):
        self.domain = domain
        self.dsl = dsl

    @property
    def trans_tokens(self, partial_program=None):
        return self.dsl.get_trans_tokens(partial_program)

    @property
    def bool_tokens(self):
        return self.dsl.get_bool_tokens()

    def distance(self, inp: Environment, out: Environment) -> float:
        """Returns the distance between two given Environments."""
        raise NotImplementedError()
