import common.tokens.robot_tokens as robot_tokens
import common.tokens.pixel_tokens as pixel_tokens
import common.tokens.string_tokens as string_tokens
from typing import List, Type

from common.tokens.abstract_tokens import Token


class DomainSpecificLanguage:
    """A class for a domain specific language"""

    def __init__(self,
                 domain_name: str,
                 bool_tokens: List[Token],
                 trans_tokens: List[Token],
                 constraints_enabled: bool = False):
        self.domain_name = domain_name
        self._bool_tokens = bool_tokens
        self._trans_tokens = trans_tokens
        self._constraints_enabled = constraints_enabled

    def get_trans_tokens(self, partial_program=None) -> List[Token]:
        """This method gets trans tokens"""

        if partial_program is None:
            partial_program = []

        return self._trans_tokens

    def get_bool_tokens(self) -> List[Token]:
        """This method gets bool tokens"""

        return self._bool_tokens

    def enable_constraints(self):
        """Enables whether constraints are accounted for in this domain specific language"""

        self._constraints_enabled = True

    def disable_constraints(self):
        """disables whether constraints are accounted for in this domain specific language"""

        self._constraints_enabled = False

    def toggle_constraints(self):
        """Toggles whether constraints are accounted for in this domain specific language"""

        self._constraints_enabled = not self._constraints_enabled

    def __str__(self):
        """Converts a domain specific language to a string"""

        return str(self._bool_tokens + self._trans_tokens)


class StandardDomainSpecificLanguage(DomainSpecificLanguage):

    def __init__(self, domain_name):
        if domain_name == "robot":
            super().__init__(domain_name, list(robot_tokens.BoolTokens), list(robot_tokens.TransTokens))
        elif domain_name == "pixel":
            super().__init__(domain_name, list(pixel_tokens.BoolTokens), list(pixel_tokens.TransTokens))
        elif domain_name == "string":
            super().__init__(domain_name, list(string_tokens.BoolTokens), list(string_tokens.TransTokens))
        else:
            raise NotImplementedError("this domain is not implemented, check whether you are using either "
                                      "\"robot\", \"pixel\" or \"string\" as domain_name")
