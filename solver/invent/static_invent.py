from common.program_synthesis.dsl import DomainSpecificLanguage
from common.tokens.abstract_tokens import TransToken, BoolToken, InventedToken, PatternApplicationToken, FunctionVariableToken, PatternToken
from common.tokens.control_tokens import If, LoopWhileThen
from solver.invent.invent import Invent


class StaticInvent(Invent):

    def setup(self, dsl: DomainSpecificLanguage):
        super().setup(dsl)
        self.ifs = self._all_ifs()
        self.loops = self._all_loops()
        self.perms = self._all_permutations()
        self.design_patterns = self._all_design_patterns()

    def _all_ifs(self) -> list[If]:
        res = []

        for cond in self._bool_tokens:
            if str(cond.__class__).__contains__("Not"):
                continue

            for e1 in self._trans_tokens:
                res.append(If(cond, [e1], []))

                for e2 in self._trans_tokens:

                    # Ifs with equal branches don't make sense
                    if e1 == e2:
                        continue

                    res.append(If(cond, [e1], [e2]))
        return res

    def _all_loops(self):
        res = []

        #for cond in self._bool_tokens:
            #for lb in self._trans_tokens:
                #res.append(LoopWhileThen(cond, [lb], []))
        i = 0
        for cond in self._bool_tokens:
            for lb in self._trans_tokens:
                for t1 in self._trans_tokens:
                    if t1 not in self._dsl.get_trans_tokens():
                    # if t1 not in self._dsl.get_trans_tokens([lb]):
                        i += 1
                        continue
                    res.append(LoopWhileThen(cond, [lb], [t1]))
                    #res.append(LoopWhileThen(cond, [lb, t1], []))

                    for cond1 in self._bool_tokens:
                        if str(cond.__class__).__contains__("Not"):
                            continue

                        if lb == t1 or cond == cond1:
                            continue

                        ##res.append(LoopWhileThen(cond, [If(cond1, [lb], [t1])], []))

        # print(f'loops removed: {i}')
        return res

    def _all_permutations(self):
        res = []

        i = 0
        for t1 in self._trans_tokens:
            res.append(InventedToken([t1]))

        for t1 in self._trans_tokens:
            for t2 in self._trans_tokens:
                if t2 not in self._dsl.get_trans_tokens():
                # if t2 not in self._dsl.get_trans_tokens(t1):
                    i+=1
                    continue
                res.append(InventedToken([t1, t2]))
        # print(f'permutations removed: {i}')
        return res

    def _all_design_patterns(self):
        res = []

        for p in self._pattern_tokens:
            for t in self._trans_tokens:
                res.append(PatternApplicationToken(p, t))
        # add also loops and ifs?
        return res
