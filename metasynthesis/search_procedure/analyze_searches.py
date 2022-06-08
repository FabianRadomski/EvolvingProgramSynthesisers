import matplotlib.pyplot as plt
import numpy as np

from solver.runner import runner
from solver.runner.algorithms import dicts
from solver.runner.runner import Runner
from solver.search.implementations.a_star import AStar
from solver.search.implementations.brute import Brute
from solver.search.implementations.genetic_programming import GeneticProgramming
from solver.search.implementations.large_neighborhood_search import LNS
from solver.search.implementations.metropolis import MetropolisHasting
from solver.search.implementations.my_mcts.mcts import MCTS
from solver.search.search_algorithm import SearchAlgorithm

"""
This file contain functions that can be used to analyze the behavior of single search procedures.
"""

algs = ["Brute", "LNS", "MH", "AS", "GP"]


def analyze_search_times(search: str, setting: str, test_size: str):
    """
    Returns the mean and the std of the number of iterations an algorithm performs on the full test set.
    """
    runner = Runner(dicts(), search, setting, test_size, 10, debug=False, store=False, multi_thread=True)
    _, _, stats_list = runner.run()
    execution_times = [stats["execution_time"] for stats in stats_list]

    name = search
    mean = np.mean(execution_times)
    std = np.std(execution_times)
    max_val = max(execution_times)
    min_val = min(execution_times)
    print("Algorithm: " + name)
    print("Iterations number list: " + str(execution_times))
    print("std: " + str(np.std(execution_times)))
    print("mean: " + str(np.mean(execution_times)))

    plot_iteration_hist(execution_times, name)

    return name, mean, std, max_val, min_val

def plot_iteration_hist(iterations, name):
    fig, ax = plt.subplots()

    ax.hist(iterations, bins="auto")
    ax.set_xlabel("Execution Time")
    ax.set_ylabel("Number of programs")
    ax.set_title("Number of iterations for " + name + " search procedure")

    plt.show()

def plot_times(names, avgs, stds, max_vals, min_vals, domain):
    """
    Plots the distributions of the number of iterations.
    """
    x_pos = np.arange(len(names))
    # Build the plot
    fig, ax = plt.subplots()
    ax.bar(x_pos, avgs, yerr=stds, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Average Execution Time [s]')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names)
    ax.set_title(f'Execution Times of Different Search Methods in {domain} Domain')
    ax.yaxis.grid(True)
    ax.scatter(x_pos, max_vals, color="red")
    ax.scatter(x_pos, min_vals, color="green")
    # Save the figure and show
    plt.tight_layout()
    plt.savefig(f'average_times_{domain}.png')
    plt.show()


if __name__ == "__main__":
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
    plot_times(names, means, stds, max_vals, min_vals, "Robot Path-Planning")
