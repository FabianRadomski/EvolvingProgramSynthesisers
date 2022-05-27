from common.environment.pixel_environment import PixelEnvironment
from common.program_synthesis.dsl import StandardDomainSpecificLanguage
from common.settings.settings import Settings


class PixelEntailment(Settings):

    def __init__(self):
        super().__init__("pixel", StandardDomainSpecificLanguage('pixel'))

    def distance(self, inp: PixelEnvironment, out: PixelEnvironment) -> float:
        if self.dist_fun is None:
            return 0 if inp.pixels == out.pixels else 1
        else:
            return self.dist_fun(inp, out)
