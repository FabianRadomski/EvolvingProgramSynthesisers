from typing import Callable, List, Tuple
from common.tokens.abstract_tokens import InvalidTransition, Token
from common.program import Program
from common.experiment import Example, TestCase
from search.abstract_search import SearchAlgorithm
from common.tokens.control_tokens import If, LoopIterationLimitReached, LoopWhile
import random
import math

from search.search_result import SearchResult

class Mutation():
    def __init__(self, name: str, fun: Callable[[Program], Program]):
        self.name = name
        self.fun : Callable[[Program], Program] = fun

    # Doesn't change the input argument, since the callback never modifies the original Program!
    def apply(self, program: Program) -> Program:
        return self.fun(program)

class MetropolisHasting(SearchAlgorithm):
    def __init__(self, time_limit_sec: float, iterations_limit: int = 0, best_program: Program = Program([]) ):
        super().__init__(time_limit_sec=time_limit_sec, iterations_limit=iterations_limit, best_program=best_program)

    def setup(self, examples: List[Example], trans_tokens, bool_tokens):
        self.number_of_explored_programs = 0
        self.number_of_iterations = 0
        self.cost_per_iteration = []
        self.cost = 100
        self.proposal_distribution = ProposalDistribution()
        fac = MutationFactory()
        self.proposal_distribution.add_mutation(fac.add_random_token(trans_tokens), 10)
        self.proposal_distribution.add_mutation(fac.remove_random_token(), 20)
        self.proposal_distribution.add_mutation(fac.add_loop(bool_tokens, trans_tokens), 10)
        self.proposal_distribution.add_mutation(fac.add_if_statement(bool_tokens, trans_tokens), 10)
        self.proposal_distribution.add_mutation(fac.start_over(), 2)
        self.first_iteration = True

    def iteration(self, examples: List[Example], trans_tokens, bool_tokens) -> bool:
        self.number_of_iterations += 1
        self.number_of_explored_programs += 1
        mut: Mutation = self.proposal_distribution.sample()
        self._best_program, newcost, solved = MetropolisHasting.maybe_apply_mutation(examples, self._best_program, self.cost, mut, self.first_iteration)
        self.first_iteration = False
        if(newcost != self.cost):
            self.cost_per_iteration.append((self.number_of_iterations, newcost))
        self.cost = newcost
        return not solved


    def extend_result(self, search_result: SearchResult):
        return super().extend_result(search_result)


    @staticmethod
    def maybe_apply_mutation(examples: List[Example], old_program: Program, ocost: int, mut: Mutation, first_iteration=False) -> Tuple[Program, int,
                                                                                                                                   int] :
        if not first_iteration:
            new_program = mut.apply(old_program)
        else:
            new_program = old_program
        try:
            cost = 0
            
            for case in examples:
                nenv = new_program.interp(case.input_environment)
                cost += abs(nenv.distance(case.output_environment))
            solved = False
            
            if cost < 0.1:
                solved = True
                for case in examples:
                    nenv = new_program.interp(case.input_environment)
                    solved = solved and nenv.correct(case.output_environment)
         
            #TODO change formula a bit

            alpha = 1.2
            ratio = math.exp(-alpha * cost)/math.exp(-alpha * ocost)
            if ratio > 1:
                return new_program, cost, solved
            if random.random() < ratio:
                return new_program, cost, solved
            return old_program, ocost, False
            
        except(InvalidTransition, LoopIterationLimitReached):
            return old_program, ocost, False

class ProposalDistribution():
    def __init__(self):
        self.mutations: List[Mutation] = []

    def add_mutation(self, mut: Mutation, pro: int):
        self.mutations.append((mut, pro))

    def sample(self) -> Mutation:
        # get total probability
        tot = 0
        for _, pro in self.mutations:
            tot += pro
        choice = random.randrange(tot)

        for mut, pro in self.mutations:
            if choice < pro:
                return mut
            choice -= pro

# The operation must never be allowed to modify the Program that is passed in!
class MutationFactory():
    def __init__(self):
        pass
        

    def add_random_token(self, trans_tokens) -> Mutation:
        def operation(pro: Program) -> Program:
            rand_token = random.choice(list(trans_tokens))
            return Program(pro.sequence + [rand_token])
        return Mutation("Append random token to the end of the program", operation)

    
    def remove_random_token(self) -> Mutation:
        def operation(pro: Program) -> Program:
            length = len(pro.sequence)
            if length == 0:
                return pro
            idk = random.randrange(length)
            return Program(pro.sequence[:idk] + pro.sequence[idk+1:])
        return Mutation("Remove random token of the end of the program", operation)

    
    def add_loop(self, bool_tokens, trans_tokens) -> Mutation:
        def operation(pro: Program) -> Program:

            rand_bool = random.choice(list(bool_tokens))
            rand_token = random.choice(list(trans_tokens))
            return Program(pro.sequence + [LoopWhile(rand_bool, [rand_token])])
        return Mutation("Add random loop to the end of the program", operation)
    
    
    def add_if_statement(self, bool_tokens, trans_tokens) -> Mutation:
        def operation(pro: Program) -> Program:

            rand_bool = random.choice(list(bool_tokens))
            rand_token = random.choice(list(trans_tokens))
            rand_token2 = random.choice(list(trans_tokens))
            return Program(pro.sequence + [If(rand_bool, [rand_token], [rand_token2])])
        return Mutation("Add random if to the end of the program", operation)

    
    def start_over(self) -> Mutation:
        def operation(_: Program) -> Program:
            return Program([])
        return Mutation("Start over", operation)    