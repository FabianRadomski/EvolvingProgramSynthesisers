from unittest import TestCase
from common.environment import RobotEnvironment
from common.program_synthesis.dsl import StandardDomainSpecificLanguage
from common.tokens.robot_tokens import MoveUp, MoveDown, MoveLeft, MoveRight, Grab, Drop
from metasynthesis.language_constraints.constraints.Constraints import CompleteConstraint, PartialConstraint
from metasynthesis.language_constraints.properties.PropertyFactory import PropertyFactory
from
from metasynthesis.language_constraints.constraints.ConstraintFactory import ConstraintFactory


class TestRobotConstraintFactory(TestCase):

    def setUp(self) -> None:
        dsl = StandardDomainSpecificLanguage("robot")
        env1 = RobotEnvironment(5, 0, 0, 0, 0, False)
        env2 = RobotEnvironment(5, 0, 0, 0, 0, True)
        env3 = RobotEnvironment(5, 1, 0, 1, 1, False)
        env4 = RobotEnvironment(5, 2, 2, 1, 1, False)
        tests = [env1, env2, env3, env4]
        property_types = [Identity, Independent]
        self.factory = ConstraintFactory(PropertyFactory(dsl, property_types, tests))

    def contains(self, p, q, _const, constraints):
        return _const([[p,q],[q,p]]) in constraints

    def testDefiniteProperties(self):
        props = self.factory.create()

        self.assertTrue(self.contains(MoveLeft(), MoveRight(), CompleteConstraint, props))
        self.assertTrue(self.contains(MoveUp(), MoveDown(), CompleteConstraint, props))
        self.assertTrue(self.contains(Grab(), Drop(), CompleteConstraint, props))

        self.assertTrue(self.contains(MoveLeft(), MoveUp(), PartialConstraint, props))
        self.assertTrue(self.contains(MoveLeft(), MoveRight(), PartialConstraint, props))
        self.assertTrue(self.contains(MoveLeft(), MoveDown(), PartialConstraint, props))
        self.assertTrue(self.contains(MoveUp(), MoveDown(), PartialConstraint, props))
        self.assertTrue(self.contains(MoveUp(), MoveRight(), PartialConstraint, props))
        self.assertTrue(self.contains(MoveDown(), MoveRight(), PartialConstraint, props))

    def testDefiniteNonProperties(self):
        props = self.factory.create()
        self.assertFalse(self.contains(MoveLeft(), Grab(), CompleteConstraint, props))
        self.assertFalse(self.contains(MoveUp(), Drop(), CompleteConstraint, props))
        self.assertFalse(self.contains(Grab(), MoveRight(), CompleteConstraint, props))