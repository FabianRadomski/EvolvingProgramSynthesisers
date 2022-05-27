from unittest import TestCase
from common.environment.robot_environment import RobotEnvironment
from common.program_synthesis.dsl import StandardDomainSpecificLanguage
from common.tokens.robot_tokens import MoveUp, MoveDown, MoveLeft, MoveRight, Grab, Drop
from metasynthesis.language_constraints.properties.BinaryProperties import Identity, Independent
from metasynthesis.language_constraints.properties.PropertyFactory import PropertyFactory


class TestRobotPropertyFactory(TestCase):

    def setUp(self) -> None:
        dsl = StandardDomainSpecificLanguage("robot")
        env1 = RobotEnvironment(5, 0, 0, 0, 0, False)
        env2 = RobotEnvironment(5, 0, 0, 0, 0, True)
        env3 = RobotEnvironment(5, 1, 0, 1, 1, False)
        env4 = RobotEnvironment(5, 2, 2, 1, 1, False)
        tests = [env1, env2, env3, env4]
        property_types = [Identity, Independent]
        self.factory = PropertyFactory(dsl, property_types, tests)

    def contains(self, p, q, _prop, props):
        return _prop([p,q]) in props or _prop([q, p]) in props

    def testDefiniteProperties(self):
        props = self.factory.create()
        self.assertTrue(self.contains(MoveLeft(), MoveRight(), Identity, props))
        self.assertTrue(self.contains(MoveUp(), MoveDown(), Identity, props))
        self.assertTrue(self.contains(Grab(), Drop(), Identity, props))

        self.assertTrue(self.contains(MoveLeft(), MoveUp(), Independent, props))
        self.assertTrue(self.contains(MoveLeft(), MoveRight(), Independent, props))
        self.assertTrue(self.contains(MoveLeft(), MoveDown(), Independent, props))
        self.assertTrue(self.contains(MoveUp(), MoveDown(), Independent, props))
        self.assertTrue(self.contains(MoveUp(), MoveRight(), Independent, props))
        self.assertTrue(self.contains(MoveDown(), MoveRight(), Independent, props))

    def testDefiniteNonProperties(self):
        props = self.factory.create()
        self.assertFalse(self.contains(MoveLeft(), Grab(), Identity, props))
        self.assertFalse(self.contains(MoveUp(), Drop(), Identity, props))
        self.assertFalse(self.contains(Grab(), MoveRight(), Identity, props))
