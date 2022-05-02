import enum
import itertools
from typing import List, Type, Dict, Iterable

from common.environment import Environment
from common.tokens.abstract_tokens import Token
from common.program_synthesis.dsl import DomainSpecificLanguage
from metasynthesis.language_constraints.properties.AbstractProperties import AbstractProperty, PropertyType, \
    WhileAbstractProperty, \
    IfAbstractProperty, TransitiveAbstractProperty, BoolAbstractProperty

Properties = List[Type[AbstractProperty]]
Input = Environment
Output = Environment
TestCases = List[Input]


class PropertyFactory:

    def __init__(self, dsl: DomainSpecificLanguage, property_types: Properties, test_case: TestCases):
        self.dsl = dsl
        self.property_types = property_types
        self.test_cases = test_case

    def _constraint_dsl(self, prop: Type[AbstractProperty]) -> List[Token]:
        if issubclass(prop, BoolAbstractProperty):
            return self.dsl.get_bool_tokens()
        if issubclass(prop, TransitiveAbstractProperty):
            return self.dsl.get_trans_tokens()

    def _match_property(self, prop: Type[AbstractProperty]):
        if prop.property_type() == PropertyType.UNARY:
            return 1
        if prop.property_type() == PropertyType.BINARY:
            return 2
        if prop.property_type() == PropertyType.TERNARY:
            return 3

    def _combine_dsl(self, dsl_subset: List[Token], size: int) -> Iterable[List[Token]]:
        return map(lambda x: list(x), itertools.combinations(dsl_subset, size))

    def _validated_properties(self, properties):
        for test_case in self.test_cases:
            filterList = []
            for prop in properties:
                filterList.append(prop.holds_for(test_case))
            properties: Iterable[AbstractProperty] = list(itertools.compress(properties, filterList))
        return properties

    def create(self) -> List[AbstractProperty]:
        properties = []
        for property_type in self.property_types:
            usable_dsl_for_property = self._constraint_dsl(property_type)
            combinations = self._combine_dsl(usable_dsl_for_property, self._match_property(property_type))
            for tokens in combinations:
                prop = property_type(tokens)
                if prop.relevant():
                    properties.append(prop)

        return list(filter(lambda prop: not prop.absurd, self._validated_properties(properties)))
