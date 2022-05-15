import matplotlib.pyplot as plt
import numpy as np

from common.program_synthesis.runner import Runner
from search.a_star.a_star import AStar
from search.abstract_search import SearchAlgorithm
from search.brute.brute import Brute
from search.MCTS.mcts import MCTS
from search.metropolis_hastings.metropolis import MetropolisHasting
from search.vlns.large_neighborhood_search.algorithms.remove_n_insert_n import RemoveNInsertN

"""
This file contain functions that can be used to analyze the behavior of single search procedures.
"""

algs = [Brute, MCTS, MetropolisHasting, RemoveNInsertN, AStar]


def analyze_search_iterations(search: SearchAlgorithm):
    """
    Returns the mean and the std of the number of iterations an algorithm performs on the full test set.
    """
    runner: Runner = Runner(search_method=search, MULTI_PROCESS=True)
    search_results: dict = runner.run()['programs']
    iterations = [search_result.dictionary['number_of_iterations'] for search_result in search_results]

    name = str(search.__class__.__name__)
    mean = np.mean(iterations)
    std = np.std(iterations)
    print("Algorithm: " + name)
    print("Iterations number list: " + str(iterations))
    print("std: " + str(np.std(iterations)))
    print("mean: " + str(np.mean(iterations)))

    plot_iteration_hist(iterations, name)

    return name, mean, std

def plot_iteration_hist(iterations, name):
    fig, ax = plt.subplots()

    ax.hist(iterations, bins="auto")
    ax.set_xlabel("Number of iterations")
    ax.set_ylabel("Number of programs")
    ax.set_title("Number of iterations for " + name + " search procedure")

    plt.show()

def plot_iterations(names, avgs, means):
    """
    Plots the distributions of the number of iterations.
    """
    x_pos = np.arange(len(names))
    # Build the plot
    fig, ax = plt.subplots()
    ax.bar(x_pos, means, yerr=stds, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Number of iterations')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names)
    ax.set_title('Average number of iterations')
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    plt.savefig('average_iter.png')
    plt.show()


if __name__ == "__main__":
    names = []
    means = []
    stds = []
    for alg in algs:
        name, mean, std = analyze_search_iterations(alg(10))
        names.append(name)
        means.append(mean)
        stds.append(std)
