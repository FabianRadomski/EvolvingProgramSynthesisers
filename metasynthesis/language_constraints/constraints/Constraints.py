import itertools
from copy import deepcopy
from typing import Type, List
from collections import defaultdict
from common.tokens.abstract_tokens import Token

Sequence = List[Token]
Constraints = List[Token]


class AbstractConstraint:

    def __init__(self, constraints: List[Constraints]):
        self.constraints = constraints
        self.tokens = []
        for tokens in constraints:
            self.tokens += tokens
        self.tokens = set(self.tokens)
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

        return set(tokens) <= self.tokens

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

    def get_values(self):
        "returns the number of values this constraint can take for use in a genetic algorithm"
        raise NotImplementedError()

    def set_value(self, index: int):
        "sets the value of this constraint can take for use in a genetic algorithm"
        raise NotImplementedError()

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v))
        return result

    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self.tokens) + ")"


class PartialConstraint(AbstractConstraint):

    def __init__(self, constraints):
        super().__init__(constraints)
        if len(constraints) > 0:
            self._allowed_constraint = constraints[0]
        else:
            self._allowed_constraint = []

    def set_partial_constraint(self, allowed_constraint=None, index=-1) -> bool:
        """
        A partial constraint is a constraint that only allows one of several sequences. This function allows for a
        change in what function is allowed.
        """
        if allowed_constraint is not None and allowed_constraint in self.constraints:
            self._allowed_constraint = allowed_constraint
            return True

        elif -1 < index < len(self._allowed_constraint):
            self._allowed_constraint = self.constraints[index]
            return True
        return False

    def constraint(self, seq: Sequence) -> Constraints:
        if len(seq) == 0 or not self.enabled:
            return []
        if self._check_constraint_relevance(seq):
            return self._allowed_constraint[0:self._allowed_constraint.index(seq[-1])]
        return []

    def get_values(self):
        return len(self.constraints) + 1

    def set_value(self, index: int):
        if index == 0:
            self.disable()
        else:
            self.enable()
            self.set_partial_constraint(index=(index - 1))
        return self


class CompleteConstraint(AbstractConstraint):

    def __init__(self, constraints):
        super(CompleteConstraint, self).__init__(constraints)

    def constraint(self, seq: Sequence) -> Constraints:
        if len(seq) == 0 or not self.enabled:
            return []
        if self._check_constraint_relevance(seq):
            return [c for c in self.constraints[0] if c is not seq[-1]]
        return []

    def get_values(self):
        return 2

    def set_value(self, index: int):
        if index == 0:
            self.disable()
        if index == 1:
            self.enable()
        return self


class MixedConstraint(AbstractConstraint):

    def __init__(self, partial_constraints, complete_constraints):
        self.partial = PartialConstraint(partial_constraints)
        self.complete = complete_constraints
        self.tokens = self.partial.tokens.union(CompleteConstraint(complete_constraints).tokens)
        super().__init__([])

    def _get_relevant_sequence(self, seq):
        relevant = []
        for token in seq:
            if token in self.tokens:
                relevant.append(token)
            else:
                relevant = []
        return relevant

    def get_values(self):
        return self.partial.get_values()

    def set_value(self, index: int):
        if index == 0:
            self.disable()
        else:
            self.partial.set_value(index)
            self.enable()
        return self

    def constraint(self, seq: Sequence) -> Constraints:
        if not self.enabled or len(seq) == 0:
            return []
        relevant_sequence = self._get_relevant_sequence(seq)
        constraints = set(self.partial.constraint(seq))

        for token in set(relevant_sequence):
            for complete_constraint in self.complete:
                if token in complete_constraint:
                    constraints.union(set(complete_constraint).difference({token}))

        return list(constraints)

    def __repr__(self):
        return self.__class__.__name__ + "(Partial["+str(self.partial.tokens)+"], Complete["+str(CompleteConstraint(self.complete).tokens)+"])"
