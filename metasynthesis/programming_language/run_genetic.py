from metasynthesis.programming_language.evolving_language import EvolvingLanguage
from evaluation.experiment_procedure import extract_bool_tokens_from_domain_name
from evaluation.experiment_procedure import extract_trans_tokens_from_domain_name

from common.program_synthesis.dsl import StandardDomainSpecificLanguage

domain_name = "robot"
# domain_name = "pixel"
# domain_name = "string"

bool_tokens = list(extract_bool_tokens_from_domain_name(domain_name))
trans_tokens = list(extract_trans_tokens_from_domain_name(domain_name))

dsl = StandardDomainSpecificLanguage("robot")


genetic = EvolvingLanguage(1, 100, 0.5, 0.5, 5, dsl)

genetic.run_evolution()

