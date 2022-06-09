from metasynthesis.language_constraints.results_parsing.Data_Reader import DataReader


def runtime_evaluation(data):
    normal_dict = {}
    constraint_dict = {}
    for time in data:
        normal = data[time]['normal']
        constraint = data[time]['constraints']
        normal_dict[time] = []
        constraint_dict[time] = []
        for trial in normal:
            for program in trial:
                normal_dict[time].append(program["execution_time"])
        for trial in constraint:
            for program in trial:
                constraint_dict[time].append(program["execution_time"])
    return normal_dict, constraint_dict

if __name__ == '__main__':
    dr = DataReader('pixelASPG')
    data = dr.get_evaluation_data()
    norm, cons = runtime_evaluation(data)
    print(sum(norm['10'])/len(norm['10']), sum(cons['10'])/len(cons['10']))