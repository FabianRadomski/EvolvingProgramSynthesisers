import json
import os
import re


class DataReader:

    def __init__(self, directory):
        self.path = os.path.join(os.path.dirname(__file__), "../results", directory)

    def get_chromosome_data(self):
        chromosomes = {}
        for file in os.scandir(self.path):
            if file.is_file() and (s := re.search(r'([0-9]+)\.json', file.path)):
                chromosome = s.group(1)
                with open(file.path, 'r') as f:
                    chromosomes[chromosome] = json.load(f)
        return chromosomes

    def get_genetic_data(self):
        for file in os.scandir(self.path):
            if file.is_file() and (s := re.search(r'genetic_run_statistics', file.path)):
                with open(file.path, 'r') as f:
                    data = json.load(f)
        return data

    def get_evaluation_data(self):
        evaluation = {}
        for file in os.scandir(self.path):
            if file.is_file() and (s := re.search(r'[0-9]+_BEST_.*_.*_(.*)_.*\.json', file.path)):
                eval_time = s.group(1)
                with open(file.path, 'r') as f:
                    evaluation[eval_time] = json.load(f)
        return evaluation

if __name__ == '__main__':
    dr = DataReader('pixelASPG')
    dr.get_evaluation_data()