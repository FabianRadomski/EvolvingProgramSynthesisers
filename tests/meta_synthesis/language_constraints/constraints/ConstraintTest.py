from unittest import TestCase
from common.environment.robot_environment import RobotEnvironment
from common.tokens.robot_tokens import MoveUp, MoveDown, MoveLeft, MoveRight, Grab, Drop
from metasynthesis.language_constraints.constraints.Constraints import CompleteConstraint, PartialConstraint, AbstractConstraint


class TestRobotConstraint(TestCase):

    def setUp(self) -> None:
        self.tokens = [MoveUp(), MoveDown(), MoveLeft(), MoveRight(), Grab(), Drop()]
        self.used = [MoveUp(), MoveRight()]
        self.con1 = PartialConstraint([MoveUp(), MoveRight()])
        self.con2 = CompleteConstraint([MoveUp(), MoveRight()])

    def test_constraint(self):
        for token in self.tokens:
            if token in self.used:
                continue
            self.assertEqual(self.con1.constraint([token]), set())
            self.assertEqual(self.con2.constraint([token]), set())
        print(self.con1.perms)
        self.assertEqual(self.con1.constraint([MoveUp()]), set())
        self.assertEqual(self.con1.constraint([MoveRight()]), {MoveUp()})
        self.assertEqual(self.con2.constraint([MoveUp()]), {MoveRight()})
        self.assertEqual(self.con2.constraint([MoveRight()]), {MoveUp()})

        self.con1.active = 1
        self.assertEqual(self.con1.constraint([MoveUp()]), {MoveRight()})
        self.assertEqual(self.con1.constraint([MoveRight()]), set())

