from common.environment.string_environment import StringEnvironment
from common.settings.settings import Settings
from common.tokens.string_tokens import TransTokens, BoolTokens


class StringEntailment(Settings):

    def __init__(self):
        super().__init__("string", TransTokens, BoolTokens)

    def distance(self, inp: StringEnvironment, out: StringEnvironment) -> float:
        if self.dist_fun is None:
            return 0 if inp.string_array == out.string_array else 1
        else:
            return self.dist_fun(inp, out)
