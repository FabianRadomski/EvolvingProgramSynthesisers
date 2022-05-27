import enum
from typing import List, Tuple

from common.environment.environment import Environment
from common.tokens.abstract_tokens import Token, BoolToken, TransToken, ControlToken, InvalidTransition
from common.tokens.control_tokens import If, LoopWhile

TokenSequence = List[Token]
Input = Environment

class PropertyType(enum.Enum):
    UNARY = enum.auto(),
    BINARY = enum.auto(),
    TERNARY = enum.auto(),

class AbstractProperty:

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.absurd = True

    def relevant(self) -> bool:
        """determines wether a property is relevant for it's given tokens"""
        raise NotImplementedError()

    def holds_for(self, input: Input) -> bool:
        """determines wether a property holds for a certain test-case"""
        try:
            h = self.hypothesize(input)
            self.absurd = False
        except InvalidTransition:
            return True
        return h

    def hypothesize(self, input: Input) -> bool:
        """determines how to test the property tokens using the input"""
        raise NotImplementedError()

    def derive_constraint(self):
        """determines how a constraint derived from a property is instantiated"""
        raise NotImplementedError()

    @classmethod
    def property_type(self):
        """Returns the type of the property"""
        raise NotImplementedError()

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.tokens == other.tokens

    def __repr__(self):
        return str(self.__class__.__name__) + "(" + str(self.tokens) + ")"

class BoolAbstractProperty(AbstractProperty):

    def relevant(self) -> bool:
        return sum(map(lambda x: isinstance(x, BoolToken), self.tokens)) == len(self.tokens)


class TransitiveAbstractProperty(AbstractProperty):

    def relevant(self) -> bool:
        return sum(map(lambda x: isinstance(x, TransToken), self.tokens)) == len(self.tokens)


class IfAbstractProperty(AbstractProperty):

    def relevant(self) -> bool:
        return len(self.tokens) == 1 and isinstance(self.tokens[0], If)


class WhileAbstractProperty(AbstractProperty):

    def relevant(self) -> bool:
        return len(self.tokens) == 1 and isinstance(self.tokens, LoopWhile)
