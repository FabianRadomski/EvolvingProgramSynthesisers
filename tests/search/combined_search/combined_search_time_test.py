import unittest

from common.program_synthesis.runner import Runner
from search.a_star.a_star import AStar
from search.brute.brute import Brute
from search.combined_search.combined_search import CombinedSearch
from search.combined_search.combined_search_time import CombinedSearchTime
from search.MCTS.mcts import MCTS
from search.metropolis_hastings.metropolis import MetropolisHasting
from search.vlns.large_neighborhood_search.algorithms.remove_n_insert_n import RemoveNInsertN


class CombinedSearchTests(unittest.TestCase):

    def run_combined(self, alg_seq):
        search = CombinedSearchTime(0, alg_seq)
        runner = Runner(search_method=search)
        results = runner.run()
        print("combined: " + str(alg_seq) + " with success rate " + str(results['average_success']) + "\n")
        return results['average_success']

    def run_solo(self, alg, time):
        search = alg(time)
        runner = Runner(search_method=search)
        results = runner.run()
        print("solo: " + str(alg.__name__) + " with success rate " + str(results['average_success']) + "\n")
        return results['average_success']

    def compare_algs(self, alg1, t1, alg2, t2):
        success_combined = self.run_combined([(alg1, t1), (alg2, t2)])
        success1 = self.run_solo(alg1, t1)
        success2 = self.run_solo(alg2, t2)
        self.assertTrue(success_combined >= success1)
        self.assertTrue(success_combined >= success2)

    def test_combination1(self):
        self.compare_algs(Brute, 0.1, MetropolisHasting, 0.2)

    def test_combination2(self):
        self.compare_algs(AStar, 0.4, Brute, 0.2)

    def test_combination3(self):
        self.compare_algs(Brute, 0.1, AStar, 0.3)

    def test_combination4(self):
        self.compare_algs(MetropolisHasting, 0.3, RemoveNInsertN, 0.5)

    def test_combination5(self):
        self.compare_algs(RemoveNInsertN, 0.1, MCTS, 0.2)

    def test_longer_chain(self):
        success_combined = self.run_combined([(Brute, 0.3), (AStar, 0.3), (MetropolisHasting, 0.2), (RemoveNInsertN, 0.1), (MCTS, 0.4)])
        success1 = self.run_solo(Brute, 0.3)
        success2 = self.run_solo(AStar, 0.3)
        success3 = self.run_solo(MetropolisHasting,  0.2)
        success4 = self.run_solo(RemoveNInsertN, 0.1)
        success5 = self.run_solo(MCTS, 0.4)
        self.assertTrue(success_combined >= success1)
        self.assertTrue(success_combined >= success2)
        self.assertTrue(success_combined >= success3)
        self.assertTrue(success_combined >= success4)
        self.assertTrue(success_combined >= success5)


if __name__ == '__main__':
    unittest.main()