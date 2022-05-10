import unittest

from common.program_synthesis.runner import Runner
from search.a_star.a_star import AStar
from search.brute.brute import Brute
from search.combined_search.combined_search import CombinedSearch
from search.MCTS.mcts import MCTS
from search.metropolis_hastings.metropolis import MetropolisHasting
from search.vlns.large_neighborhood_search.algorithms.remove_n_insert_n import RemoveNInsertN


class CombinedSearchTests(unittest.TestCase):

    def run_test(self, alg_seq):
        search = CombinedSearch(0, alg_seq)
        runner = Runner(search_method=search, print_results=True)
        results = runner.run()
        print(results)

    def test_combination1(self):
        self.run_test([(Brute, 10), (MetropolisHasting, 10)])

    def test_combination2(self):
        self.run_test([(AStar, 10), (Brute, 10)])

    def test_combination3(self):
        self.run_test([(Brute, 10), (AStar, 10)])

    def test_combination4(self):
        self.run_test([(MetropolisHasting, 10), (RemoveNInsertN, 1000)])

    def test_combination5(self):
        self.run_test([(RemoveNInsertN, 1000), (MCTS, 1000)])

    def test_longer_chain(self):
        self.run_test([(Brute, 3), (AStar, 3), (MetropolisHasting, 3), (RemoveNInsertN, 300), (MCTS, 300)])


if __name__ == '__main__':
    unittest.main()
