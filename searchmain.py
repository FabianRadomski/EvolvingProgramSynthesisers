import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from metasynthesis.search_procedure.search_synthesiser import SearchSynthesiser

fig_save = {"SO": "strings", "PO": "pixels", "RO": "robots"}
fig_name = {"SO": "String", "PO": "Pixel", "RO": "Robot"}


def plot_fitness(fitnesses):
    markers = ['o', '^', 's', 'x']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, fitness in enumerate(fitnesses):
        x = np.arange(1, len(fitness) + 1)
        plt.plot(x, fitness, marker=markers[i], label=f"$p_m = {mutation_probabilities[i]}$")
    plt.xlabel("Generation")
    plt.ylabel("Average Fitness")
    plt.title(f"Fitness over Generations in the {fig_name[setting]} Domain")
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    plt.savefig(f"fit-{fig_save[setting]}.png")
    plt.show()


def plot_successes(evolutions_history, mutations_results):
    markers = ['o', '^', 's', 'x']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, calculated_results in enumerate(mutations_results):
        success_rates_gens = []
        evolution_history = evolutions_history[i]
        for gen in evolution_history.values():
            success_rates = []
            for ind, fit in gen:
                success_rates.append(calculated_results[tuple(ind)]['average_success'])
            success_rates_gens.append(np.mean(success_rates))
        x = np.arange(1, len(success_rates_gens) + 1)
        ax.plot(x, success_rates_gens, marker=markers[i], label=f"$p_m = {mutation_probabilities[i]}$")
    plt.xlabel("Generation")
    plt.ylabel("Average Success Rate")
    plt.title(f"Average Success Rate over Generations in the {fig_name[setting]} Domain\n")
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    plt.savefig(f"succ-{fig_save[setting]}.png")
    plt.show()


def plot_speeds(speeds):
    markers = ['o', '^', 's', 'x']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, speed in enumerate(speeds):
        x = np.arange(1, len(speed) + 1)
        plt.plot(x, speed, marker=markers[i], label=f"$p_m = {mutation_probabilities[i]}$")
    plt.xlabel("Generation")
    plt.ylabel("Time")
    plt.title(f"Average Execution Time of Search Procedures in the {fig_name[setting]} Domain")
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    plt.savefig(f"exec-{fig_save[setting]}.png")
    plt.show()

def calculate_avg_fitness(generations):
    print(generations)
    fit_mean = []
    for i in range(len(generations[0])):
        fit = []
        for gen_list in generations:
            fit.append(np.mean([fitness for genotype, fitness in gen_list[i]]))
        fit_mean.append(np.mean(fit))
    return fit_mean

def calculate_avg_speed(speeds):
    avg_speed = []
    for i in range(len(speeds[0])):
        avg_speed.append(np.mean([speed[i] for speed in speeds]))
    return avg_speed

if __name__ == "__main__":
    mutation_probabilities = [0.01, 0.05, 0.1, 0.2]

    mutations_results = []
    fitnesses = []
    avg_speeds = []
    setting = "RO"
    for mut in mutation_probabilities:
        hists, speeds = [], []
        for i in range(3):
            ss = SearchSynthesiser(fitness_limit=0, generation_limit=100, crossover_probability=0.8,
                                   mutation_probability=mut, generation_size=50, max_seq_size=6, dist_type="Time", print_generations=True,
                                   setting=setting, test_size="param", plot=True, write_generations=True)
            hist, results, speed = ss.run_evolution()
            hists.append(list(hist.values()))
            # results2.append(results)
            speeds.append(speed)
        fitnesses.append(calculate_avg_fitness(hists))
        #mutations_results.append(np.mean(results2)[0])
        #evolutions_history.append(np.mean(results2)[0])
        avg_speeds.append(calculate_avg_speed(speeds))

    #plot_successes(evolutions_history, mutations_results)
    plot_speeds(avg_speeds)
    plot_fitness(fitnesses)

# if __name__ == "__main__":
#     mutation_probabilities = [0.01, 0.05, 0.1, 0.2]
#
#     mutations_results = []
#     evolutions_history = []
#     avg_speeds = []
#     setting = "RO"
#     for mut in mutation_probabilities:
#         ss = SearchSynthesiser(fitness_limit=0, generation_limit=50, crossover_probability=0.8,
#                                mutation_probability=mut, generation_size=100, max_seq_size=6, dist_type="Time", print_generations=True,
#                                setting=setting, test_size="param", plot=True, write_generations=True)
#         hist, results, speed = ss.run_evolution()
#         mutations_results.append(results)
#         evolutions_history.append(hist)
#         avg_speeds.append(speed)
#     plot_successes(evolutions_history, mutations_results)
#     plot_speeds(avg_speeds)
#     plot_fitness(evolutions_history)