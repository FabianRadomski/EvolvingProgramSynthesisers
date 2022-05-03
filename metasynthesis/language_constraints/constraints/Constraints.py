import itertools
from typing import Type, List
from collections import defaultdict
from common.tokens.abstract_tokens import Token

Sequence = List[Token]
Constraints = List[Token]


class AbstractConstraint:

    def __init__(self, constraints: List[Constraints]):
        self._constraints = constraints
        self.enabled = True

    def _get_relevant_sequence(self, seq: Sequence) -> Sequence:
        """returns the relevant subsequence for the given constraint"""
        if len(seq) > 0:
            return [seq[-1]]
        else:
            return []

    def _check_constraint_relevance(self, seq: Sequence) -> bool:
        """returns wether the given token sequence is relevant for this constraint"""
        if len(seq) == 0:
            return False
        tokens = self._get_relevant_sequence(seq)

        for i, token in enumerate(reversed(tokens)):
            if not token == seq[-i]:
                break
        else:
            return True

        return False

    def disable(self) -> None:
        """disables the constraint"""
        self.enabled = False

    def enable(self) -> None:
        """enables the constraint"""
        self.enabled = True

    def toggle_enable(self) -> None:
        """toggles whether the constraint is enabled"""
        self.enabled = not self.enabled

    def constraint(self, seq: Sequence) -> Constraints:
        """returns a list of tokens to be constrained for the given input"""
        raise NotImplementedError()


class PartialConstraint(AbstractConstraint):

    def __init__(self, constraints):
        super().__init__(constraints)
        if len(constraints) > 0:
            self._allowed_constraint = constraints[0]
        else:
            self._allowed_constraint = []
        self._constraint_dict = self._initialize_constraint_dict()

    def _initialize_constraint_dict(self):
        """Initializes a memoised default dict for returning constraint tokens"""
        constraint_dict = {}
        tokens = set(list(itertools.chain(*self._constraints)))
        for token in tokens:
            constraint_dict[token] = []
            for constraint in self._constraints:
                if not token in constraint or constraint[-1] == token or constraint == self._allowed_constraint:
                    continue
                # add the next constraint in sequence
                constraint_dict[token].append(constraint[constraint.index(token) + 1])
        return defaultdict(lambda: [], constraint_dict)

    def set_partial_constraint(self, allowed_constraint=None, index=-1) -> bool:
        """
        A partial constraint is a constraint that only allows one of several sequences. This function allows for a
        change in what function is allowed.
        """
        if allowed_constraint is not None and allowed_constraint in self._constraints:
            self._allowed_constraint = allowed_constraint
            self._constraint_dict = self._initialize_constraint_dict()
            return True

        elif -1 < index < len(self._allowed_constraint):
            self._allowed_constraint = self._constraints[index]
            self._constraint_dict = self._initialize_constraint_dict()
            return True
        return False

    def constraint(self, seq: Sequence) -> Constraints:
        if len(seq) == 0 or not self.enabled:
            return []
        if self._check_constraint_relevance(seq):
            return self._constraint_dict[seq[-1]]
        return []



class CompleteConstraint(AbstractConstraint):

    def __init__(self, constraints):
        super(CompleteConstraint, self).__init__(constraints)
        self._constraint_dict = self._initialize_constraint_dict()

    def _initialize_constraint_dict(self):
        """Initializes a memoised default dict for returning constraint tokens"""
        constraint_dict = {}
        tokens = set(list(itertools.chain(*self._constraints)))
        for token in tokens:
            constraint_dict[token] = []
            for constraint in self._constraints:
                if not token in constraint or constraint[-1] == token:
                    continue
                # add the next constraint in sequence
                constraint_dict[token].append(constraint[constraint.index(token) + 1])
        return defaultdict(lambda: [], constraint_dict)

    def constraint(self, seq: Sequence) -> Constraints:
        if len(seq) == 0 or not self.enabled:
            return []
        if self._check_constraint_relevance(seq):
            return self._constraint_dict[seq[-1]]
        return []

