import random

from common.program_synthesis.dsl import DomainSpecificLanguage
from common.tokens.abstract_tokens import TransToken, BoolToken, EnvToken
from common.tokens.control_tokens import If, LoopWhile


class Invent:

    def __init__(self):
        self._trans_tokens = []
        self._bool_tokens = []

        self.loops = []
        self.ifs = []
        self.perms = []

    def setup(self, dsl: DomainSpecificLanguage):
        self._dsl = dsl
        self._trans_tokens = dsl.get_trans_tokens()
        self._bool_tokens = dsl.get_bool_tokens()

    def increment_depth(self):
        pass


    def random_token(self, w_trans: float, w_if: float, w_loop: float) -> EnvToken:
        return random.choices([
            self.random_trans_token,
            self.random_if_token,
            self.random_loop_token,
        ], [w_trans, w_if, w_loop], k=1)[0]()

    def random_trans_token(self) -> TransToken:
        return random.sample(self._trans_tokens, 1)[0]

    def random_if_token(self) -> If:
        return random.sample(self.ifs, 1)[0]

    def random_loop_token(self) -> LoopWhile:
        return random.sample(self.loops, 1)[0]