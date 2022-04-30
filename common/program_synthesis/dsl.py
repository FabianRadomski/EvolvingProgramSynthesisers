import common.tokens.robot_tokens as robot_tokens
from typing import List, Type

from common.tokens.abstract_tokens import Token


class DomainSpecificLanguage:
    """A class for a domain specific language"""

    def __init__(self, bool_tokens: List[Type[Token]], trans_tokens: List[Type[Token]],
                 constraints_enabled: bool = False):
        self._bool_tokens = bool_tokens
        self._trans_tokens = trans_tokens
        self._constraints_enabled = constraints_enabled

    def get_trans_tokens(self, partial_program=None) -> List[Type[Token]]:
        """This method gets trans tokens"""

        if partial_program is None:
            partial_program = []

        return self._trans_tokens

    def get_bool_tokens(self):
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

class StandardRobotDomainSpecificLanguage(DomainSpecificLanguage):
    
    def __init__(self):
        super(StandardRobotDomainSpecificLanguage, self).__init__(robot_tokens.BoolTokens, robot_tokens.TransTokens)