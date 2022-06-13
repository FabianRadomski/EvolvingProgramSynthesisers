import itertools
from typing import List

from common.tokens.abstract_tokens import Token


class AbstractConstraint:

    def __init__(self, tokens):
        self.state = 0
        self.tokens = sorted(tokens)
        self.token_set = set(tokens)

    def check_token(self, token):
        """takes state and an input tokens, returns False if the input token can't be added"""
        raise NotImplementedError()

    def update_state(self, token):
        """updates the state of the constraint with the given input token"""
        if token in self.get_constraint():
            raise InvalidSequenceException()

    def constraint(self, tokens: List[Token]):
        for token in tokens:
            self.update_state(token)
        constraints = self.get_constraint()
        self.state = 0
        return constraints

    def get_constraint(self):
        """gets the constained tokens from the given state"""
        raise NotImplementedError()

    def state_values(self):
        raise NotImplementedError()

    def get_values(self):
        "returns the number of values this constraint can take for use in a genetic algorithm"
        raise NotImplementedError()

    def set_value(self, index: int):
        "sets the value of this constraint can take for use in a genetic algorithm"
        raise NotImplementedError()

    @property
    def active(self):
        return self._active

    @active.setter
    def active(self, index):
        self._active = index


    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self.tokens) + ")"


class PartialConstraint(AbstractConstraint):

    def __init__(self, tokens):
        super().__init__(tokens)
        self.perms = list(itertools.permutations(tokens))
        self._active = 0

    def check_token(self, token):
        """takes state and an input tokens, returns False if the input token can't be added"""
        if token not in self.tokens:
            return True
        if token in self.get_constraint():
            return False
        return True

    def get_constraint(self):
        allowed = {token for i, token in enumerate(self.perms[self.active]) if i >= self.state-1}
        return {token for token in self.perms[self.active] if token not in allowed}

    def update_state(self, token):
        super(PartialConstraint, self).update_state(token)
        if token in self.perms[self.active]:
            self.state = self.perms[self.active].index(token)+1
        else:
            self.state = 0

    def state_values(self):
        return len(self.perms[self.active])

    def get_values(self):
        return len(self.perms)

    def set_value(self, index: int):
        if index >= len(self.perms):
            self.active = len(self.perms) - 1
            return self
        self.active = index
        return self

    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self.perms[self.active]) + ")"

class CompleteConstraint(AbstractConstraint):

    def __init__(self, tokens):
        self._active = 1
        self.dcs = set()
        super().__init__(tokens)

    def set_dc_values(self, dcs):
        self.dcs = dcs

    def check_token(self, token):
        return token not in self.tokens or self.state == 0

    def update_state(self, token):
        super(CompleteConstraint, self).update_state(token)
        if token in self.tokens:
            self.state = self.tokens.index(token)+1
            return
        if token in self.dcs:
            return
        else:
            self.state = 0

    def get_constraint(self):
        if self.state == 0:
            return set()
        if self.state == (len(self.tokens)+1):
            return self.token_set
        else:
            return {t for t in self.tokens if t is not self.tokens[self.state-1]}

    def state_values(self):
        return len(self.tokens)+1

    def get_values(self):
        return 1

    def set_value(self, index: int):
        self.active = index
        return self

class InvalidSequenceException(Exception):
    pass
