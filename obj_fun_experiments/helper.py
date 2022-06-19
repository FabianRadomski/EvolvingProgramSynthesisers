import ast
import os

import numpy as np
import matplotlib.pyplot as plt


def get_max_fit_values(path):
    with open(path) as f:
        for line in f.readlines():
            arr = line.split('=')
            if arr[0] == 'best_fit':
                return np.array(ast.literal_eval(arr[1]))


def get_avg_fit_values(path):
    with open(path) as f:
        for line in f.readlines():
            arr = line.split('=')
            if arr[0] == 'avg_fit':
                return np.array(ast.literal_eval(arr[1]))


def get_all_params(path):
    params = dict()
    with open(path) as f:
        for line in f.readlines():
            arr = line.split('=')
            params[arr[0]] = arr[1].split('\n')[0]
    return params


def config_to_best_eval(path):
    params = []
    with open(path) as f:
        for line in f.readlines():
            arr = line.split('=')
            if arr[0] == 'avg_fit':
                break
            params.append(line.split('\n')[0])
    return {tuple(params): get_eval_best(path)}


def get_config(i, path):
    res = str(i) + ' & '
    with open(path) as f:
        for line in f.readlines():
            arr = line.split('=')
            if arr[0] in ['domain', 'w1', 'w2', 'w3']:
                continue
            if arr[0] == 'avg_fit':
                break
            res += arr[1].split('\n')[0] + ' & '
    res = res[:-2]
    res += '\\\\ \hline'
    return res


def get_config_robot(i, path):
    res = str(i) + ' & '
    with open(path) as f:
        for line in f.readlines():
            arr = line.split('=')
            if arr[0] in ['domain', 'w1', 'w2', 'w3', 'time_out', 'pop_size']:
                continue
            if arr[0] == 'avg_fit':
                break
            res += arr[1].split('\n')[0] + ' & '
    res = res[:-2]
    res += '\\\\ \hline'
    return res


def get_eval_best(path):
    params = dict()
    with open(path) as f:
        for line in f.readlines():
            arr = line.split('=')
            if arr[0] == 'evaluation_best':
                return round(float(arr[1]), 3)


def manually_defined(path):
    params = dict()
    with open(path) as f:
        for line in f.readlines():
            arr = line.split('=')
            if arr[0] == 'evaluation_manually_designed':
                return round(float(arr[1]), 3)


def plot_max_fit_values(path):
    max_fit_values = get_max_fit_values(path)
    xdata = np.arange(max_fit_values.shape[0]) + 1
    plt.plot(xdata, max_fit_values, 'o--')
    plt.show()


def plot_avg_fit_values(path):
    avg_fit_values = get_avg_fit_values(path)
    xdata = np.arange(avg_fit_values.shape[0]) + 1
    plt.plot(xdata, avg_fit_values, 'o--')
    plt.show()


def plot_avg_vs_best(path):
    best = get_max_fit_values(path)
    avg = get_avg_fit_values(path)
    xdata = np.arange(best.shape[0]) + 1
    plt.plot(xdata, best, 'o--')
    plt.plot(xdata, avg, 'o--')
    plt.show()


def plot_n_avg(paths):
    shapes = ['*', 'o', 'v', 'h', 'D']
    i = 0
    for path in paths:
        avg = get_avg_fit_values(path)
        xdata = np.arange(avg.shape[0]) + 1
        plt.plot(xdata, avg, shapes[i] + '--', markersize=9)
        i += 1 % len(shapes)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Fitness', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()


def plot_n_best(paths):
    shapes = ['*', 'o', 'v', 'h', 'D']
    i = 0
    for path in paths:
        best = get_max_fit_values(path)
        xdata = np.arange(best.shape[0]) + 1
        plt.plot(xdata, best, shapes[i] + '--', markersize=9)
        i += 1 % len(shapes)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Fitness', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()


def res_table_entry(i, path):
    res = str(i) + ' & '
    res += str(round(get_max_fit_values(path)[-1], 3)) + ' & '
    res += str(get_eval_best(path))
    # res += str(get_eval_best(path)) + ' & '
    # res += str(manually_defined(path))
    res += '\\\\ \hline'
    return res


def configuration_table(root):
    # configuration table
    i = 1
    cur = ''
    prev = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            s = get_config(i=i, path=os.path.join(path, name))
            if cur == '':
                print(s)
                cur = s
                prev.append(s.split('&')[1:])
            else:
                if s.split('&')[1:] in prev:
                    continue
                else:
                    print(s)
                    cur = s
                    prev.append(s.split('&')[1:])
            i += 1


def configuration_table_robot(root):
    # configuration table
    i = 1
    cur = ''
    prev = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            s = get_config_robot(i=i, path=os.path.join(path, name))
            if cur == '':
                print(s)
                cur = s
                prev.append(s.split('&')[1:])
            else:
                if s.split('&')[1:] in prev:
                    continue
                else:
                    print(s)
                    cur = s
                    prev.append(s.split('&')[1:])
            i += 1


def results_table(root):
    # results table
    i = 1
    cur = ''
    prev = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            s = get_config(i=i, path=os.path.join(path, name))
            if cur == '':
                print(res_table_entry(i, os.path.join(path, name)))
                cur = s
                prev.append(s.split('&')[1:])
            else:
                if s.split('&')[1:] in prev:
                    continue
                else:
                    print(res_table_entry(i, os.path.join(path, name)))
                    cur = s
                    prev.append(s.split('&')[1:])
            i += 1


def get_entry(i, path):
    res = str(i) + ' & '
    with open(path) as f:
        for line in f.readlines():
            arr = line.split('=')
            if arr[0] in ['domain', 'w1', 'w2', 'w3', 'time_out', 'pop_size']:
                continue
            if arr[0] == 'avg_fit':
                break
            res += arr[1].split('\n')[0] + ' & '
    res = res[:-2]
    res += '& ' + str(round(get_max_fit_values(path)[-1], 3)) + ' & '
    res += str(get_eval_best(path))
    res += '\\\\ \hline'
    return res


def table(root):
    # configuration table
    i = 1
    cur = ''
    prev = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            s = get_entry(i=i, path=os.path.join(path, name))
            if cur == '':
                print(s)
                cur = s
                prev.append(s.split('&')[1:8])
            else:
                if s.split('&')[1:8] in prev:
                    continue
                else:
                    print(s)
                    cur = s
                    prev.append(s.split('&')[1:8])
            i += 1


if __name__ == '__main__':
    table(root='robot/')
    table(root='string/')

    root = 'robot/new param tournament_size/'
    plot_n_best(paths=[root+'3_1', root+'3_2', root+'3_3', root+'3_4', root+'3_5'])
    plot_n_avg(paths=[root + '3_1', root + '3_2', root + '3_3', root + '3_4', root + '3_5'])
