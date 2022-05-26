from typing import Tuple, List

from common.environment.environment import Environment
from common.program import Program
from common.tokens.abstract_tokens import Token, InvalidTransition
from metasynthesis.language_constraints.properties.AbstractProperties import BoolAbstractProperty, \
    TransitiveAbstractProperty, \
    PropertyType
from metasynthesis.language_constraints.constraints.Constraints import PartialConstraint, CompleteConstraint

TokenSequence = List[Token]
Input = Environment
Output = Environment
TestCase = Tuple[TokenSequence, Input]


class Exclusive(BoolAbstractProperty):

    def relevant(self) -> bool:
        return len(self.tokens) == 2 and super().relevant()

    def hypothesize(self, input):
        p = self.tokens[0]
        q = self.tokens[1]
        return p.apply(input) != q.apply(input)

    def derive_constraint(self):
        pass

    @classmethod
    def property_type(self):
        return PropertyType.BINARY


class Independent(TransitiveAbstractProperty):

    def relevant(self) -> bool:
        return len(self.tokens) == 2 and super().relevant()

    def hypothesize(self, input) -> bool:
        p = self.tokens[0]
        q = self.tokens[1]
        return p.apply(q.apply(input.__deepcopy__())) == q.apply(p.apply(input.__deepcopy__()))

    def derive_constraint(self):
        p = self.tokens[0]
        q = self.tokens[1]

        return PartialConstraint([[p, q], [q, p]])

    @classmethod
    def property_type(self):
        return PropertyType.BINARY


class Identity(TransitiveAbstractProperty):

    def relevant(self) -> bool:
        return len(self.tokens) == 2 and super().relevant()

    def hypothesize(self, input) -> bool:
        p = self.tokens[0]
        q = self.tokens[1]
        return p.apply(q.apply(input.__deepcopy__())) == input

    def derive_constraint(self):
        p = self.tokens[0]
        q = self.tokens[1]

        return CompleteConstraint([[p, q], [q, p]])

    @classmethod
    def property_type(self):
        return PropertyType.BINARY
