from dataclasses import dataclass, field
# takes as input a DSL,
from typing import Type

from common.program_synthesis.dsl import DomainSpecificLanguage
from search.abstract_search import SearchAlgorithm


@dataclass
class Runner:
    """Runner for running a program synthesizer for a given domain specific language NOT FOR a meta-synthesizer"""

    search_method: Type[SearchAlgorithm]
    dsl: DomainSpecificLanguage
    domain_name: str
    files: field(default_factory=tuple)


