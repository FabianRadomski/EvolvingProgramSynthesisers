from common.program_synthesis.dsl import StandardDomainSpecificLanguage
from common.settings.settings import Settings
from common.environment.pixel_environment import PixelEnvironment


class PixelHamming(Settings):
    """Hamming settings for pixel environments."""

    def __init__(self):
        super().__init__("pixel", StandardDomainSpecificLanguage('pixel'))

    def distance(self, inp: PixelEnvironment, out: PixelEnvironment) -> float:
        assert len(inp.pixels) == len(out.pixels)
        return sum([e1 != e2 for (e1, e2) in zip(inp.pixels, out.pixels)])