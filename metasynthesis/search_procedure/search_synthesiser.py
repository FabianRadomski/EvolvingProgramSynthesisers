# TESTING SERACH SEQUENCES
from search.a_star.a_star import AStar
from search.batch_run import BatchRun
from search.brute.brute import Brute
from search.MCTS.mcts import MCTS
from search.metropolis_hastings import metropolis
from search.metropolis_hastings.metropolis import MetropolisHasting
from search.vlns.large_neighborhood_search.algorithms.remove_n_insert_n import RemoveNInsertN


class SearchSynthesiser:

    def seq_test(self):
        alg_seq = [(MetropolisHasting, 1000), (MCTS, 1000)]
        results = BatchRun(
            # Task domain
            domain="robot",

            # Iterables for files name. Use [] to use all values.
            # This runs all files adhering to format "2-*-[0 -> 10]"
            # Thus, ([], [], []) runs all files for a domain.
            files=([], [], []),

            # Search algorithm to be used
            search_algorithm=Brute(10),

            # Prints out result when a test case is finished
            print_results=True,

            # Use multi core processing
            multi_core=True,

            # Use file_name= to append to a file whenever a run got terminated
            # Comment out argument to create new file.
            # file_name="VLNS-20211213-162128.txt"
        ).run_seq(alg_seq)
        # print(result)


if __name__ == "__main__":
    SearchSynthesiser().seq_test()
