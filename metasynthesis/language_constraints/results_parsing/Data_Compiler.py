import itertools
from statistics import mean

from metasynthesis.language_constraints.results_parsing.Data_Reader import DataReader
import pandas as pd

def _evaluation(data, keys):
    normal_dict = {}
    constraint_dict = {}
    for time in data:
        normal = data[time]['normal']
        constraint = data[time]['constraints']
        normal_dict[time] = []
        constraint_dict[time] = []
        for trial in normal:
            for program in trial:
                t = []
                for key in keys:
                    t.append(program[key])
                normal_dict[time].append(tuple(t))
        for trial in constraint:
            for program in trial:
                t = []
                for key in keys:
                    t.append(program[key])
                constraint_dict[time].append(tuple(t))
    return normal_dict, constraint_dict

def cost_evaluation(data):
    norm, cons = _evaluation(data, ["complexity", "test_cost"])
    def parse(data, key):
        return [(g, sum(map(lambda k: k[1], k))) for g, k in itertools.groupby(sorted(data[key], key=lambda x: x[0]), lambda x: x[0])]
    norm = {k: parse(norm, k) for k in norm}
    cons = {k: parse(cons, k) for k in cons}
    print("normal: ", norm)
    print("constrainted: ", cons)
    return norm, cons

def accuracy_evaluation(data):
    norm, cons = _evaluation(data, ["complexity", "test_correct"])
    def parse(data, key):
        return [(g, mean(map(lambda k: k[1], k))) for g, k in itertools.groupby(sorted(data[key], key=lambda x: x[0]), lambda x: x[0])]
    norm = {k: parse(norm, k) for k in norm}
    cons = {k: parse(cons, k) for k in cons}
    print("normal: ", norm)
    print("constrainted: ", cons)
    return norm, cons

def programs_evaluation(data):
    norm, cons = _evaluation(data, ["complexity", "no._explored_programs"])
    def parse(data, key):
        return [(g, mean(map(lambda k: k[1], k))) for g, k in itertools.groupby(sorted(data[key], key=lambda x: x[0]), lambda x: x[0])]
    norm = {k: parse(norm, k) for k in norm}
    cons = {k: parse(cons, k) for k in cons}
    print(norm)
    print(cons)
    return norm, cons

def runtime_evaluation(data):
    norm, cons = _evaluation(data, ["complexity", "execution_time"])
    def parse(data, key):
        return [(g, mean(map(lambda k: k[1], k))) for g, k in itertools.groupby(sorted(data[key], key=lambda x: x[0]), lambda x: x[0])]
    norm = {k: parse(norm, k) for k in norm}
    cons = {k: parse(cons, k) for k in cons}
    print(norm)
    print(cons)
    return norm, cons