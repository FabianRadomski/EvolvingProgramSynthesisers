import itertools
from functools import reduce
from typing import List, Set

import networkx

from common.environment.robot_environment import RobotEnvironment
from common.program_synthesis.dsl import StandardDomainSpecificLanguage
from common.tokens.abstract_tokens import Token
from common.tokens.robot_tokens import MoveDown
from metasynthesis.language_constraints.constraints.Constraints import AbstractConstraint, PartialConstraint, \
    CompleteConstraint, MixedConstraint
from metasynthesis.language_constraints.properties.BinaryProperties import Independent, Identity
from metasynthesis.language_constraints.properties.PropertyFactory import PropertyFactory
import networkx as nx


class ConstraintFactory:

    # TODO create property_factory internally using same data constructor
    def __init__(self, property_factory: PropertyFactory):
        self._property_factory = property_factory

    def create(self) -> List[AbstractConstraint]:
        """creates a list of constraints using a given PropertyFactory"""
        properties = self._property_factory.create()
        return list(map(lambda prop: prop.derive_constraint(), properties))


class ConstraintCombiner:

    def __init__(self, constraints, trans_tokens: List[Token]):
        self._constraints = constraints
        self._trans_tokens = trans_tokens

    def combine_all(self):
        partial_constraints = list(filter(lambda c: isinstance(c, PartialConstraint), self._constraints))
        complete_constraints = list(filter(lambda c: isinstance(c, CompleteConstraint), self._constraints))
        return self._combine_partial(partial_constraints, complete_constraints)

    def combine(self, constraint_a: AbstractConstraint, constraint_b: AbstractConstraint):
        if isinstance(constraint_a, PartialConstraint) and isinstance(constraint_b, PartialConstraint):
            final_tokens: Set[Token] = constraint_a.token_set.union(constraint_b.token_set)
            return PartialConstraint(final_tokens)

    def _combine_partial(self, partial_constraints: List[PartialConstraint], complete_constraints: List[CompleteConstraint]):
        G = networkx.Graph()
        G.add_nodes_from(self._trans_tokens)
        G.add_edges_from([c.token_set for c in partial_constraints])

        cliques = list(nx.find_cliques(G))
        constraint_cliques_p = [list(filter(lambda t: t.token_set <= set(clique), partial_constraints)) for clique in cliques]
        constraint_cliques_c = [list(filter(lambda t: t.token_set <= set(clique), complete_constraints)) for clique in cliques]
        left_over_complete = [constraint for constraint in complete_constraints if constraint not in [c for clique in constraint_cliques_c for c in clique]]
        left_over_partial = [constraint for constraint in partial_constraints if constraint not in [c for clique in constraint_cliques_p for c in clique]]

        final_constraints = []

        for clique_p, clique_c in zip(constraint_cliques_p, constraint_cliques_c):
            tokens = set()
            for constraint in clique_p:
                tokens = tokens.union(constraint.token_set)

            if clique_p and clique_c:
                final_constraints.append(MixedConstraint(tokens, clique_c))


        return final_constraints+left_over_partial+left_over_complete




if __name__ == '__main__':
    dsl = StandardDomainSpecificLanguage("robot")
    env1 = RobotEnvironment(5, 0, 0, 0, 0, False)
    env2 = RobotEnvironment(5, 0, 0, 0, 0, True)
    env3 = RobotEnvironment(5, 1, 0, 1, 1, False)
    env4 = RobotEnvironment(5, 2, 2, 1, 1, False)
    tests = [env1, env2, env3, env4]
    property_types = [Identity, Independent]
    factory = ConstraintFactory(PropertyFactory(dsl, property_types, tests))
    constraints = factory.create()
    #print(constraints)
    tokens = dsl.get_trans_tokens()
    combiner = ConstraintCombiner(constraints, tokens)
    c = combiner.combine_all()
    print(c)
