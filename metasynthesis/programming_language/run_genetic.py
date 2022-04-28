from metasynthesis.programming_language.evolving_language import EvolvingLanguage
from evaluation.experiment_procedure import extract_bool_tokens_from_domain_name
from evaluation.experiment_procedure import extract_trans_tokens_from_domain_name

domain_name = "robot"

all_tokens = list(extract_bool_tokens_from_domain_name(domain_name)
                  .union(extract_trans_tokens_from_domain_name(domain_name)))

genetic = EvolvingLanguage(1, 100, 0.5, 0.5, domain_name, all_tokens)

genetic.run_evolution()

