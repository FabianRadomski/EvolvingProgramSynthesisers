import common.tokens.robot_tokens as robot_tokens
import common.tokens.pixel_tokens as pixel_tokens
import common.tokens.string_tokens as string_tokens
from typing import List, Type, Callable

from common.tokens.abstract_tokens import Token

ConstraintFunc = Callable[[List[Token]], List[Token]]

class DomainSpecificLanguage:
    """A class for a domain specific language"""

    def __init__(self,
                 domain_name: str,
                 bool_tokens: List[Type[Token]],
                 trans_tokens: List[Type[Token]],
                 constraints_enabled: bool = False,
                 constraint_func: ConstraintFunc = lambda x: x):
        self.domain_name = domain_name
        self._bool_tokens = bool_tokens
        self._trans_tokens = trans_tokens
        self._constraints_enabled = constraints_enabled
        self._constraint_func = constraint_func

    def get_trans_tokens(self, partial_program=None) -> List[Type[Token]]:
        """This method gets trans tokens"""
        if not self._constraints_enabled:
            return self._trans_tokens
        if partial_program is None:
            partial_program = []
        return self._constraint_func[partial_program]

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


class StandardDomainSpecificLanguage(DomainSpecificLanguage):

    def __init__(self, domain_name):
        if domain_name == "robot":
            super().__init__(domain_name, robot_tokens.BoolTokens, robot_tokens.TransTokens)
        elif domain_name == "pixel":
            super().__init__(domain_name, pixel_tokens.BoolTokens, pixel_tokens.TransTokens)
        elif domain_name == "string":
            super().__init__(domain_name, string_tokens.BoolTokens, string_tokens.TransTokens)
        else:
            raise NotImplementedError("this domain is not implemented, check whether you are using either "
                                      "\"robot\", \"pixel\" or \"string\" as domain_name")
