from copy import deepcopy
from typing import List

from common.tokens.abstract_tokens import TransToken, EnvToken, InventedToken
from common.tokens.control_tokens import If, LoopWhile, LoopWhileThen
from metasynthesis.language_constraints.constraints.Constraints import AbstractConstraint, InvalidSequenceException


class Constraint_Walker:

    def __init__(self, constraints: List[AbstractConstraint]):
        self.constraints = constraints

    def __call__(self, token: EnvToken):
        if isinstance(token, InventedToken):
            for t in token.tokens:
                try:
                    [c.update_state(t) for c in self.constraints]
                    return True
                except InvalidSequenceException:
                    return False
        if isinstance(token, If):
            copy = deepcopy(self.constraints)
            for t in token.e1:
                [c.update_state(t) for c in copy]
            for t in token.e2:
                [c.update_state(t) for c in self.constraints]
            self.combine(copy)
        if isinstance(token, LoopWhile):
            copy = deepcopy(self.constraints)
            for t in token.loop_body:
                [c.update_state(t) for c in copy]
            self.combine(copy)
        if isinstance(token, LoopWhileThen):
            copy = deepcopy(self.constraints)
            for t in token.loop_body:
                [c.update_state(t) for c in copy]
            for t in token.then_body:
                [c.update_state(t) for c in copy]
                [c.update_state(t) for c in self.constraints]
            self.combine(copy)

    def combine(self, copy):
        for c, f in zip(copy, self.constraints):
            f.state = max(c.state, f.state)

