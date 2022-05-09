from typing import List
from metasynthesis.language_constraints.constraints.Constraints import AbstractConstraint
from metasynthesis.language_constraints.properties.PropertyFactory import PropertyFactory


class ConstraintFactory:

    #TODO create property_factory internally using same data constructor
    def __init__(self, property_factory: PropertyFactory):
        self._property_factory = property_factory

    def create(self) -> List[AbstractConstraint]:
        """creates a list of constraints using a given PropertyFactory"""
        properties = self._property_factory.create()
        return list(map(lambda prop: prop.derive_constraint(), properties))

