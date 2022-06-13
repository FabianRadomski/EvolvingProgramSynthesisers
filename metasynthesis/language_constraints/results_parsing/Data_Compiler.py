import itertools

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

def runtime_evaluation(data):
    norm, cons = _evaluation(data, ["execution_time"])
    for key in norm:
        print(norm)
        print(f'{key}:{sum(map(lambda x: x[0], norm[key]))/len(norm[key])} normal')
        print(f'{key}:{sum(map(lambda x: x[0], cons[key])) / len(cons[key])} constrained')
    return norm, cons

def cost_evaluation(data):
    norm, cons = _evaluation(data, ["complexity", "test_cost"])
    def parse(data, key):
        return [(g, sum(map(lambda k: k[1], k))) for g, k in itertools.groupby(sorted(data[key], key=lambda x: x[0]), lambda x: x[0])]
    norm = {k: parse(norm, k) for k in norm}
    cons = {k: parse(cons, k) for k in cons}
    print("normal: ", norm)
    print("constrainted: ", cons)
    return norm, cons

if __name__ == '__main__':
    dr = DataReader('pixelASPE')
    data = dr.get_evaluation_data()
    norm, cons = cost_evaluation(data)
