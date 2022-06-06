from metasynthesis.search_procedure.analyze_searches import analyze_search_times,  plot_times

algs = ["Brute", "LNS", "MH", "AS"]

names = []
means = []
stds = []
max_vals = []
min_vals = []
for alg in algs:
    name, mean, std, max_val, min_val = analyze_search_times(alg, "RO", "param")
    names.append(name)
    means.append(mean)
    stds.append(std)
    max_vals.append(max_val)
    min_vals.append(min_val)
plot_times(names, means, stds, max_vals, min_vals)