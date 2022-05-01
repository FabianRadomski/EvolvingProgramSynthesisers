import itertools
from typing import Type, List

from common.tokens.abstract_tokens import Token

Sequence = List[Token]
Constraints = List[Token]


class AbstractConstraint:

    def __init__(self, constraints: List[Constraints]):
        self._constraints = constraints
        self.enabled = True

    def disable(self):
        self.enabled = False

    def enable(self):
        self.enabled = True

    def toggle_enable(self):
        self.enabled = not self.enabled

    def constraint(self, seq: Sequence) -> Constraints:
        raise NotImplementedError()


class PartialConstraint(AbstractConstraint):

    def __init__(self, constraints):
        super().__init__(constraints)
        if len(constraints) > 0:
            self._allowed_constraint = constraints[0]
        else:
            self._allowed_constraint = []
        self._constraint_dict = self.__initialize_constraint_dict()

    def _initialize_constraint_dict(self):
        constraint_dict = {}
        tokens = set(itertools.chain(self._constraints))
        for token in tokens:
            constraint_dict[token] = []
            for constraint in self._constraints:
                if not token in constraint or constraint[-1] == token:
                    continue
                # add the next constraint in sequence
                constraint_dict[token].append(constraint[constraint.index(token)+1])
        return constraint_dict

    def set_partial_constraint(self, allowed_constraint=None, index=-1) -> bool:
        if allowed_constraint is not None and allowed_constraint in self._constraints:
            self._allowed_constraint = allowed_constraint
            return True

        elif -1 < index < len(self._allowed_constraint):
            self._allowed_constraint = self._constraints[index]
            return True
        return False

    def constraint(self, seq: Sequence) -> Constraints:
        if len(seq) == 0:
            return []
        if self._check_constraint_relevance(seq):
            return self._constraint_dict[seq[-1]]
        return []

    def _get_relevant_sequence(self, seq: Sequence):
        if len(seq) > 0:
            return [seq[-1]]
        else:
            return []

    def _check_constraint_relevance(self, seq: Sequence) -> Bool:
        if len(seq) == 0:
            return False
        tokens = self._get_relevant_sequence(seq)

        for i, token in enumerate(reversed(tokens)):
            if not token == seq[-i]:
                break
        else:
            return True

        return False



class CompleteConstraint(AbstractConstraint):

    def __init__(self, constraints):
        super(CompleteConstraint, self).__init__(constraints)
