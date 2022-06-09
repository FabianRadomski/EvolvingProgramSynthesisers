import os
import re


class DataReader:

    def __init__(self, directory):
        self.path = os.path.join(__file__, "../results", directory)

    def get_chromosome_data(self):
        chromosomes = {}

        for file in os.scandir(self.path):
            if file.isfile() and re.search(r'[0-9]+\.json', file.path):
