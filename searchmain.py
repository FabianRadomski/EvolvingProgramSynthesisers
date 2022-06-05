from metasynthesis.search_procedure.search_synthesiser import SearchSynthesiser

ss = SearchSynthesiser(fitness_limit=0, generation_limit=50, crossover_probability=0.8,
                       mutation_probability=0.2, generation_size=20, max_seq_size=6, dist_type="Time", print_generations=True,
                       setting="RO", test_size="param", plot=True, write_generations=True)
ss.run_evolution()