from unittest import TestCase

from common.program_synthesis.dsl import StandardDomainSpecificLanguage
from common.tokens.abstract_tokens import InventedToken
from common.tokens.control_tokens import If, LoopWhile, LoopWhileThen
from common.tokens.robot_tokens import Grab, Drop, MoveUp, MoveDown, MoveLeft, MoveRight, AtTop
from metasynthesis.language_constraints.constraints.Constraints import PartialConstraint, CompleteConstraint, \
    InvalidSequenceException
from metasynthesis.language_constraints.genetic_algorithm.Constraint_Checker import ConstraintBuffer, apply_constraint
from solver.invent.static_invent import StaticInvent


class ConstraintBufferTest(TestCase):

    def setUp(self) -> None:
        self.constraints = [PartialConstraint([MoveUp(), MoveDown(), MoveLeft(), MoveRight()]), CompleteConstraint([MoveLeft(), MoveRight()]), CompleteConstraint([Grab(), Drop()])]
        self.tokens = [
                  InventedToken([Drop()]),
                  InventedToken([Grab()]),
                  InventedToken([MoveUp()]),
                  InventedToken([MoveDown()]),
                  InventedToken([MoveLeft()]),
                  InventedToken([MoveRight()]),
                  If(AtTop(),[MoveLeft()], [MoveRight()]),
                  LoopWhile(AtTop(), [MoveUp()]),
                  LoopWhile(AtTop(), [MoveUp(), MoveDown()]),
                  InventedToken([MoveDown(), MoveUp()]),
                  LoopWhileThen(AtTop(), [MoveUp()], [MoveDown()])]

        self.constraintBuffer = ConstraintBuffer(self.constraints, self.tokens)

    def test_border_calculation(self):
        for token in self.tokens:
            for constraint, state in zip(self.constraints, self.constraintBuffer._border[token]):
                if state > constraint.state_values():
                    constraint.state = constraint.state_values()
                    self.assertTrue(apply_constraint(token, constraint))
                    continue
                if state == 0:
                    constraint.state = 0
                    self.assertRaises(InvalidSequenceException, lambda: apply_constraint(token, constraint))
                    continue
                constraint.state = state
                self.assertRaises(InvalidSequenceException, lambda: apply_constraint(token, constraint))
                constraint.state = state - 1
                self.assertTrue(apply_constraint(token, constraint))

    def test_check_token(self):
        for token in self.tokens:
            for constraint, state in zip(self.constraints, self.constraintBuffer._border[token]):
                if state == 0:
                    constraint.state = 0
                    self.assertRaises(InvalidSequenceException, lambda: apply_constraint(token, constraint))
                    self.assertRaises(InvalidSequenceException, lambda: self.constraintBuffer.check_token([token]))




