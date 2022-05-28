import unittest

from solver.runner.algorithms import dicts
from solver.runner.runner import Runner


class CombinedSearchTests(unittest.TestCase):
    setting = "RE"
    test_cases = "small"

    def run_combined(self, alg_seq, setting):
        runner = Runner(dicts(alg_sequence=alg_seq), algo="CS", setting=setting, test_cases=self.test_cases, time_limit_sec=10, debug=False,
                        store=False, multi_thread=True)
        avg_success, average_time = runner.run()
        print("combined: " + str(alg_seq) + " with success rate " + str(avg_success) + "\n")
        return avg_success

    def run_solo(self, alg, time, setting):
        runner = Runner(dicts(), algo=alg, setting=setting, test_cases=self.test_cases, time_limit_sec=time, debug=False,
                        store=False, multi_thread=True)
        avg_success, average_time = runner.run()
        print("solo: " + str(alg) + " with success rate " + str(avg_success) + "\n")
        print("avg time" + str(average_time) + "\n")
        return avg_success

    def compare_algs(self, alg1, t1, alg2, t2, setting):
        success_combined = self.run_combined([(alg1, t1), (alg2, t2)], setting)
        success1 = self.run_solo(alg1, t1, setting)
        success2 = self.run_solo(alg2, t2, setting)
        self.assertTrue(success_combined >= success1)
        self.assertTrue(success_combined >= success2)

    def test_combination1(self):
        self.compare_algs("Brute", 0.3, "MH", 0.3, self.setting)

    def test_combination2(self):
        self.compare_algs("AS", 0.3, "Brute", 0.3, self.setting)

    def test_combination3(self):
        self.compare_algs("Brute", 0.3, "AS", 0.3, self.setting)

    def test_combination4(self):
        self.compare_algs("MH", 0.3, "LNS", 0.3, self.setting)

    def test_combination5(self):
        self.compare_algs("LNS", 0.3, "MCTS", 0.3, self.setting)

    def test_combination6(self):
        self.compare_algs("LNS", 0.3, "GP", 0.3, self.setting)

    def test_combination7(self):
        self.compare_algs("GP", 0.3, "Brute", 0.3, self.setting)

    def test_buggy_combination1(self):
        self.compare_algs("MCTS", 0.3, "MH", 0.3, self.setting)

    def test_buggy_combination2(self):
        self.compare_algs("MCTS", 0.3, "AS", 0.3, self.setting)

    def test_longer_chain(self):
        success_combined = self.run_combined([("Brute", 0.3), ("AS", 0.3), ("MH", 0.3), ("LNS", 0.3), ("MCTS", 0.3)], self.setting)
        success1 = self.run_solo("Brute", 0.3, self.setting)
        success2 = self.run_solo("AS", 0.3, self.setting)
        success3 = self.run_solo("MH", 0.3, self.setting)
        success4 = self.run_solo("LNS", 0.3, self.setting)
        success5 = self.run_solo("MCTS", 0.3, self.setting)
        self.assertTrue(success_combined >= success1)
        self.assertTrue(success_combined >= success2)
        self.assertTrue(success_combined >= success3)
        self.assertTrue(success_combined >= success4)
        self.assertTrue(success_combined >= success5)


if __name__ == '__main__':
    unittest.main()
