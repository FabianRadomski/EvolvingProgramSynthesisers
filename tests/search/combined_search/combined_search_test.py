import unittest

from common.program_synthesis.runner import Runner
from search.a_star.a_star import AStar
from search.brute.brute import Brute
from search.combined_search.combined_search import CombinedSearch
from search.MCTS.mcts import MCTS
from search.metropolis_hastings.metropolis import MetropolisHasting
from search.vlns.large_neighborhood_search.algorithms.remove_n_insert_n import RemoveNInsertN


class CombinedSearchTests(unittest.TestCase):

    def run_combined(self, alg_seq):
        search = CombinedSearch(0, alg_seq)
        runner = Runner(search_method=search)
        results = runner.run()
        print("combined: " + str(alg_seq) + " with success rate " + str(results['average_success']) + "\n")
        return results['average_success']

    def run_solo(self, alg, iterations):
        search = alg(0, iterations_limit=iterations)
        runner = Runner(search_method=search)
        results = runner.run()
        print("solo: " + str(alg.__name__) + " with success rate " + str(results['average_success']) + "\n")
        return results['average_success']

    def compare_algs(self, alg1, iter1, alg2, iter2):
        success_combined = self.run_combined([(alg1, iter1), (alg2, iter2)])
        success1 = self.run_solo(alg1, iter1)
        success2 = self.run_solo(alg2, iter2)
        self.assertTrue(success_combined >= success1)
        self.assertTrue(success_combined >= success2)

    def test_combination1(self):
        self.compare_algs(Brute, 10, MetropolisHasting, 50)

    def test_combination2(self):
        self.compare_algs(AStar, 10, Brute, 10)

    def test_combination3(self):
        self.compare_algs(Brute, 10, AStar, 10)

    def test_combination4(self):
        self.compare_algs(MetropolisHasting, 10, RemoveNInsertN, 1000)

    def test_combination5(self):
        self.compare_algs(RemoveNInsertN, 1000, MCTS, 1000)

    def test_longer_chain(self):
        success_combined = self.run_combined([(Brute, 3), (AStar, 3), (MetropolisHasting, 20), (RemoveNInsertN, 300), (MCTS, 300)])
        success1 = self.run_solo(Brute, 3)
        success2 = self.run_solo(AStar, 3)
        success3 = self.run_solo(MetropolisHasting, 20)
        success4 = self.run_solo(RemoveNInsertN, 300)
        success5 = self.run_solo(MCTS, 300)
        self.assertTrue(success_combined >= success1)
        self.assertTrue(success_combined >= success2)
        self.assertTrue(success_combined >= success3)
        self.assertTrue(success_combined >= success4)
        self.assertTrue(success_combined >= success5)


if __name__ == '__main__':
    unittest.main()
