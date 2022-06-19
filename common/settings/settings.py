from typing import Callable, List

from common.environment.environment import Environment
from common.tokens.abstract_tokens import Token, TransToken, BoolToken


class Settings:
    """Abstract settings class."""

    def __init__(self, domain: str, trans_tokens: List[TransToken], bool_tokens: List[BoolToken]):
        self.domain = domain
        self.trans_tokens = trans_tokens
        self.bool_tokens = bool_tokens
        self.dist_fun = None

    def distance(self, inp: Environment, out: Environment) -> float:
        """Returns the distance between two given Environments."""
        raise NotImplementedError()
