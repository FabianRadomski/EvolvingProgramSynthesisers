import itertools
from functools import reduce
from queue import PriorityQueue
from typing import List, Set

import networkx

from common.environment.pixel_environment import PixelEnvironment
from common.environment.string_environment import StringEnvironment
from common.program_synthesis.dsl import StandardDomainSpecificLanguage
from common.tokens.abstract_tokens import Token
from common.tokens.string_tokens import MakeLowercase, MakeUppercase
from metasynthesis.language_constraints.constraints.Constraints import AbstractConstraint, PartialConstraint, \
    CompleteConstraint
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

    def _combine_partial(self, partial_constraints: List[PartialConstraint],
                         complete_constraints: List[CompleteConstraint]):
        G = networkx.Graph()
        G.add_nodes_from(self._trans_tokens)
        G.add_edges_from([c.token_set for c in partial_constraints])

        cliques = list(nx.find_cliques(G))
        q = PriorityQueue()
        # remove double constraints in cliques:
        # insert in priority queue
        for clique in cliques:
            q.put((-len(clique), clique))
        seen = set()
        cliques = []
        # remove seen constraints from cliques
        # enter final cliques by order of size
        while not q.empty():
            priority, clique = q.get()
            return_to_queue = False

            for constraint in clique:
                if constraint in seen:
                    clique.remove(constraint)
                    return_to_queue = True

            if return_to_queue:
                q.put((-len(clique), clique))
            elif clique:
                for constraint in clique:
                    seen.add(constraint)
                cliques.append(clique)

        constraint_cliques_p = [list(filter(lambda t: t.token_set <= set(clique), partial_constraints)) for clique in
                                cliques]
        constraint_cliques_c = [list(filter(lambda t: t.token_set <= set(clique), complete_constraints)) for clique in
                                cliques]
        left_over_complete = [constraint for constraint in complete_constraints if
                              constraint not in [c for clique in constraint_cliques_c for c in clique]]
        left_over_partial = [constraint for constraint in partial_constraints if
                             constraint not in [c for clique in constraint_cliques_p for c in clique]]

        final_constraints = []

        for clique_p, clique_c in zip(constraint_cliques_p, constraint_cliques_c):
            tokens = set()
            value = 0
            for constraint in clique_p:
                tokens = tokens.union(constraint.token_set)

            for i, constraint in enumerate(sorted(clique_p, key=lambda c: tuple(c.tokens))[:len(tokens)]):
                value += (constraint.active) * (2**i)

            for constraint in clique_c:
                constraint.set_dc_values(tokens)
                final_constraints.append(constraint)
            if tokens:
                c = PartialConstraint(tokens)
                c.set_value(value)
                final_constraints.append(c)

        return final_constraints + left_over_partial + left_over_complete


if __name__ == '__main__':
    dsl = StandardDomainSpecificLanguage("pixel")
    env1 = PixelEnvironment(3, 3, 1, 1)
    env2 = PixelEnvironment(3, 3, 1, 1, [False, False, False, False, True, False, False, False, False])
    env3 = PixelEnvironment(3, 3, 0, 0)
    env4 = PixelEnvironment(3, 3, 0, 0, [True, True, True, True, False, True, True, True, True])
    tests = [env1, env2, env3, env4]
    property_types = [Identity, Independent]
    factory = ConstraintFactory(PropertyFactory(dsl, property_types, tests))
    constraints = factory.create()
    # print(constraints)
    tokens = dsl.get_trans_tokens()
    print(tokens)
    combiner = ConstraintCombiner(constraints, tokens)
    c = combiner.combine_all()
    print(c)
