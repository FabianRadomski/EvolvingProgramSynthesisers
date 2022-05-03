from unittest import TestCase
from common.environment import RobotEnvironment
from common.tokens.robot_tokens import MoveUp, MoveDown, MoveLeft, MoveRight, Grab, Drop
from metasynthesis.language_constraints.properties.BinaryProperties import Independent, Identity


class TestRobotProperty(TestCase):

    def setUp(self) -> None:
        self.env1 = lambda: RobotEnvironment(5, 0, 0, 0, 0, False).__deepcopy__()
        self.env2 = lambda: RobotEnvironment(5, 0, 0, 0, 0, True).__deepcopy__()
        self.env3 = lambda: RobotEnvironment(5, 1, 0, 1, 1, False).__deepcopy__()

    def test_independent(self):
        prop1 = Independent([MoveRight(), MoveRight()])
        prop2 = Independent([MoveRight(), MoveDown()])
        prop3 = Independent([MoveRight(), Drop()])
        prop4 = Independent([MoveRight(), Grab()])

        self.assertTrue(prop1.holds_for(self.env1()))
        self.assertTrue(prop1.holds_for(self.env2()))
        self.assertTrue(prop1.holds_for(self.env3()))

        self.assertTrue(prop2.holds_for(self.env1()))
        self.assertTrue(prop2.holds_for(self.env2()))
        self.assertTrue(prop2.holds_for(self.env3()))

        self.assertTrue(prop3.holds_for(self.env1()))
        self.assertFalse(prop3.holds_for(self.env2()))
        self.assertTrue(prop3.holds_for(self.env3()))

        self.assertTrue(prop4.holds_for(self.env1()))
        self.assertTrue(prop4.holds_for(self.env2()))
        self.assertTrue(prop4.holds_for(self.env3()))
        self.assertTrue(prop4.absurd)

    def test_identity(self):
        prop1 = Identity([MoveRight(), MoveRight()])
        prop2 = Identity([MoveRight(), MoveLeft()])
        prop3 = Identity([MoveRight(), Drop()])
        prop4 = Identity([MoveRight(), Grab()])

        self.assertFalse(prop1.holds_for(self.env1()))
        self.assertFalse(prop1.holds_for(self.env2()))
        self.assertFalse(prop1.holds_for(self.env3()))

        self.assertTrue(prop2.holds_for(self.env1()))
        self.assertTrue(prop2.holds_for(self.env2()))
        self.assertTrue(prop2.holds_for(self.env3()))

        self.assertTrue(prop3.holds_for(self.env1()))
        self.assertFalse(prop3.holds_for(self.env2()))
        self.assertTrue(prop3.holds_for(self.env3()))

        self.assertFalse(prop4.holds_for(self.env1()))
        self.assertTrue(prop4.holds_for(self.env2()))
        self.assertTrue(prop4.holds_for(self.env3()))

