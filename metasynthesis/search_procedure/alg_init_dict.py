from collections import defaultdict, Mapping

from solver.search.implementations.a_star import AStar
from solver.search.implementations.brute import Brute
from solver.search.implementations.genetic_programming import GeneticProgramming
from solver.search.implementations.large_neighborhood_search import LNS
from solver.search.implementations.my_mcts.mcts import MCTS
from solver.search.implementations.metropolis import MetropolisHasting

class LazyDict(Mapping):
    def __init__(self, *args, **kw):
        self._raw_dict = dict(*args, **kw)

    def __getitem__(self, key):
        func, *args = self._raw_dict.__getitem__(key)
        return func(*args)

    def __iter__(self):
        return iter(self._raw_dict)

    def __len__(self):
        return len(self._raw_dict)

def alg_init_dict(c_param = 0.5):
    return dict({"Brute": LazyDict({
        "SG": (Brute, ),
        "SO": (Brute, ),
        "SE": (Brute, ),
        "RG": (Brute, ),
        "RO": (Brute, ),
        "RE": (Brute, ),
        "PG": (Brute, ),
        "PO": (Brute, ),
        "PE": (Brute, ),
    }),
                         "AS": LazyDict({
        "SG": (AStar, 0.01),
        "SO": (AStar, 0.06),
        "SE": (AStar, 0.1),
        "RG": (AStar, 0.8),
        "RO": (AStar, 0),
        "RE": (AStar, 0.1),
        "PG": (AStar, 0),
        "PO": (AStar, 0),
        "PE": (AStar, 0.1),
    }),
    "MCTS": LazyDict({
        "SG": (MCTS, c_param, 0),
        "SO": (MCTS, c_param, 0),
        "SE": (MCTS, c_param, 0),

        "RG": (MCTS, c_param, 0),
        "RO": (MCTS, c_param, 0),
        "RE": (MCTS, c_param, 5),

        "PG": (MCTS, c_param, 0),
        "PO": (MCTS, c_param, 0),
        "PE": (MCTS, c_param, 0),

        # "SG": (MCTS, 0, 9),
        # "SO": (MCTS, 0, 5),
        # "SE": (MCTS, 0, 10),
        # "RG": (MCTS, 0, 30),
        # "RO": (MCTS, 0, 10),
        # "RE": (MCTS, 0, 10),
        # "PG": (MCTS, 0, 10),
        # "PO": (MCTS, 0, 10),
        # "PE": (MCTS, 0, 10),
    }),
    "LNS": LazyDict({
        "SG": (LNS, 4, 4),
        "SO": (LNS, 4, 4),
        "SE": (LNS, 4, 4),
        "RG": (LNS, 8, 8),
        "RO": (LNS, 2, 2),
        "RE": (LNS, 8, 8),
        "PG": (LNS, 3, 3),
        "PO": (LNS, 2, 2),
        "PE": (LNS, 3, 3),
    }),
    "MH": LazyDict({
        "SG": (MetropolisHasting, 4),
        "SO": (MetropolisHasting, 4),
        "SE": (MetropolisHasting, 4),
        "RG": (MetropolisHasting, 2),
        "RO": (MetropolisHasting, 10),
        "RE": (MetropolisHasting, 4),
        "PG": (MetropolisHasting, 4),
        "PO": (MetropolisHasting, 4),
        "PE": (MetropolisHasting, 4),
    }),
    "GP": LazyDict({
        "SG": (GeneticProgramming, 60, 0.1),
        "SO": (GeneticProgramming, 45, 0.2),
        "SE": (GeneticProgramming, 60, 0.1),
        "RG": (GeneticProgramming, 30, 0.8),
        "RO": (GeneticProgramming, 45, 0.4),
        "RE": (GeneticProgramming, 45, 0.4),
        "PG": (GeneticProgramming, 30, 0.8),
        "PO": (GeneticProgramming, 60, 0.8),
        "PE": (GeneticProgramming, 60, 0.8),
    })})
