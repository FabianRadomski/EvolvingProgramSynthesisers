from common.environment.string_environment import StringEnvironment
from common.program_synthesis.dsl import StandardDomainSpecificLanguage
from common.settings.settings import Settings


class StringEntailment(Settings):

    def __init__(self):
        super().__init__("string", StandardDomainSpecificLanguage('string'))

    def distance(self, inp: StringEnvironment, out: StringEnvironment) -> float:
        return 0 if inp.string_array == out.string_array else 1
