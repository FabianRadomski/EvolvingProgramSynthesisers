from metasynthesis.search_procedure.search_synthesiser import SearchSynthesiser

SearchSynthesiser(fitness_limit=0, generation_limit=50, crossover_probability=0.8,
                      mutation_probability=0.1, generation_size=10, max_seq_size=4, dist_type="Time", print_generations=True,
                      setting= "PO").run_evolution()