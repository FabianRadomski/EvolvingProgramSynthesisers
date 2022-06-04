import itertools
import os
from copy import deepcopy
from multiprocessing import Pool
from typing import List, Dict, Tuple, Set

from common.tokens.abstract_tokens import EnvToken, Token, TransToken, InventedToken
from common.tokens.control_tokens import LoopWhile, LoopWhileThen, If
from metasynthesis.language_constraints.constraints.Constraints import AbstractConstraint, InvalidSequenceException, \
    PartialConstraint


class ConstraintBuffer:

    def __init__(self, constraints: List[AbstractConstraint], tokens: List[EnvToken]):
        self._tokens = tokens
        self._constraints = constraints
        self._computed: Dict[Token, Dict[Tuple, Tuple]] = {}
        self._program_buffer: Dict[Tuple, Tuple] = {}
        self._invalid: Set[Tuple] = set()
        self.setup()

    def setup(self):
        for token in self._tokens:
            self._computed[token] = {}

        with Pool(processes=os.cpu_count() - 1) as pool:
            results = pool.map_async(self._setup_desync, itertools.product(self._enumerate_constraint_states(), self._tokens))

            for token, state, output in results.get():
                self._computed[token][state] = output

    def _setup_desync(self, input):
        state, token = input
        for c, v in zip(self._constraints, state):
            c.state = v
        try:
            output_state = tuple([apply_constraint(token, c).state for c in self._constraints])
            return token, state, output_state
        except InvalidSequenceException:
            return token, state, None


    def _enumerate_constraint_states(self):
        state = [range(c.state_values()+1) for c in self._constraints]
        return itertools.product(*state)

    def check_token(self, sequence: List[EnvToken]):
        if tuple(sequence) in self._program_buffer:
            return True
        if tuple(sequence) in self._invalid:
            raise InvalidSequenceException()

        # search buffer for available program state
        seq_copy = sequence[:]
        while sequence:
            sequence.pop()
            if (t := tuple(sequence)) in self._program_buffer:
                state = self._program_buffer[t]
                break
            if t in self._invalid:
                raise InvalidSequenceException()
        else:
            state = tuple([0 for _ in self._constraints])

        # build buffer back up
        to_buffer = seq_copy[len(sequence):]
        while to_buffer:
            token = to_buffer.pop(0)
            state = self._computed[token][state]
            if state is None:
                self._invalid.add(tuple(sequence + [token]))
                raise InvalidSequenceException()
            sequence.append(token)
            self._program_buffer[tuple(sequence)] = state
        return True





class Program_Buffer:

    def __init__(self, constraints):
        self._constraints = constraints
        self.buffer = {(): tuple([0 for _ in constraints])}

    def is_buffered(self, sequence):
        if (t := tuple(sequence)) in self.buffer:
            return self.buffer[t]
        return ()

    def get_state(self, sequence, token):
        buffered_sequence, constraints = self.search_buffer(sequence)
        state = self.buffered_evaluate(sequence, token, buffered_sequence, constraints)
        return state

    def buffered_evaluate(self, sequence, token, buffered_sequence=None, constraint_state=None):
        if buffered_sequence is None:
            buffered_sequence = []
        if isinstance(buffered_sequence, tuple):
            buffered_sequence = list(buffered_sequence)
        if constraint_state is None:
            constraint_state = tuple([0 for _ in self._constraints])
        sequence = sequence[len(buffered_sequence):]

        for state, constraint in zip(constraint_state, self._constraints):
            constraint.state = state

        for token in sequence:
            cs = []
            for constraint in self._constraints:
                cs.append(apply_constraint(token, constraint))
            buffered_sequence.append(token  )
            self.buffer[tuple(buffered_sequence)] = tuple([constraint.state for constraint in cs])
        cs = []
        for constraint in self._constraints:
            cs.append(apply_constraint(token, constraint))
        buffered_sequence.append(token)
        self.buffer[tuple(buffered_sequence)] = tuple([constraint.state for constraint in cs])
        return self.buffer[tuple(buffered_sequence)]

    def search_buffer(self, sequence):
        for i, _ in enumerate(sequence):
            key = tuple(sequence[:-(i - 1)])
            if key in self.buffer:
                return key, self.buffer[key]
        return None, None


def apply_constraint(token: EnvToken, constraint: AbstractConstraint) -> AbstractConstraint:
    tokens = map_tokens(token)
    constraints = []
    for ts in tokens:
        c = deepcopy(constraint)
        for token in ts:
            c.update_state(token)
        constraints.append(c)
    if len(constraints) == 1:
        return constraints[0]
    if len(constraints) == 2:
        return combine(constraints[0], constraints[1])


def map_tokens(token: EnvToken) -> List[List[TransToken]]:
    if isinstance(token, InventedToken):
        return [token.tokens]
    if isinstance(token, LoopWhile):
        return [token.loop_body + token.loop_body, []]
    if isinstance(token, LoopWhileThen):
        return [token.loop_body + token.loop_body + token.then_body, token.then_body]
    if isinstance(token, If):
        return [token.e1, token.e2]
    else:
        return [token]


def combine(constraint_a, constraint_b):
    if isinstance(constraint_a, PartialConstraint) or constraint_a.state == 0 or constraint_b.state == 0:
        constraint_a.state = max(constraint_a.state, constraint_b.state)
    elif constraint_a.state != constraint_b.state:
        constraint_a.state = len(constraint_a.tokens)+1
    return constraint_a
