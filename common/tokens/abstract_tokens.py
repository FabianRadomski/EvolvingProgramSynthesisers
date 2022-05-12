from ast import arg
from common.environment import *


class Token:
    """Abstract Token. Enforces that all tokens have an apply method."""

    def apply(self, env: Environment):
        """Applies this Token on a given Environment."""
        raise NotImplementedError()

    def number_of_tokens(self) -> int:
        return 1

    def to_formatted_string(self):
        return str(self)

    def __str__(self):
        return str(type(self).__name__)

    def __repr__(self):
        return str(type(self).__name__)

    def __eq__(self, other):
        return self.__class__ is other.__class__

    def __hash__(self):
        return hash((self.__class__))


class BoolToken(Token):
    """Abstract Token that returns a boolean value."""

    def apply(self, env: Environment) -> bool:
        """Applies this BoolToken on a given Environment. Returns a boolean value."""
        raise NotImplementedError()


class EnvToken(Token):
    """Abstract Token that returns an Environment."""

    def apply(self, env: Environment) -> Environment:
        """Applies this BoolToken on a given Environment. Returns a boolean value."""
        raise NotImplementedError()


class TransToken(EnvToken):
    """Abstract Token that can transform an Environment."""

    def apply(self, env: Environment) -> Environment:
        """Applies this TransToken on a given Environment. Alters the Environment and returns the newly obtained one."""

        raise NotImplementedError()


class ControlToken(EnvToken):
    """Abstract Token used for flow control."""

    def apply(self, env: Environment) -> Environment:
        """Applies this ControlToken on a given Environment. Alters the Environment and returns the newly obtained one."""

        raise NotImplementedError()


class InventedToken(EnvToken):
    def __init__(self, tokens: list):
        self.tokens = tokens

    def apply(self, env: Environment) -> Environment:
        for t in self.tokens:
            env = t.apply(env)

        return env

    def number_of_tokens(self) -> int:
        return sum([t.number_of_tokens() for t in self.tokens])

    def __str__(self):
        return "[%s]" % ", ".join([str(t) for t in self.tokens])

    def to_formatted_string(self):
        return "[%s]" % "\n".join([t.to_formatted_string() for t in self.tokens])

    def __repr__(self):
        return "[%s]" % ", ".join([str(t) for t in self.tokens])

# assumptions: there's only one parameter called x, we apply it by substituting all x occurences by the arg_token which is a TransToken
class FunctionDefinitionToken():
    def __init__(self, body_tokens: list, arg_token: TransToken):
        self.body_tokens = body_tokens
        self.arg_token = arg_token

class FunctionVariableToken(Token):
    def __init__(self) -> None:
        print("I'm a Variable Token, somebody is cooking up a function")
        
class FunctionApplicationToken(EnvToken):
    def __init__(self, fd: FunctionDefinitionToken, param_token: TransToken):
        self.param_token = param_token
        self.function_definition = fd

    # simple implementation with a single argument and parameter
    def apply(self, env: Environment) -> Environment:
        for token in self.function_definition.body_tokens:
            if isinstance(token, FunctionVariableToken):
                self.param_token.apply(env)
            else: 
                token.apply(env)
class InvalidTransition(Exception):
    """This exception will be raised whenever an invalid state transition is performed on an Environment."""
    pass
