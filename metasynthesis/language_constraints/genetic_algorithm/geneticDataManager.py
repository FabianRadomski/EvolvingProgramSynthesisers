import json
import os


class DataManager:

    def __init__(self, domain, algorithm, objective_function):
        dirname = f"{domain}{algorithm}{objective_function}"
        self.dir_path = os.path.join(os.path.dirname(__file__), "../results", f"{dirname}")
        print(self.dir_path)
        if not os.path.exists(self.dir_path):
            os.mkdir(self.dir_path)


    def write(self, file_name, data):
        with open(os.path.join(self.dir_path, file_name), 'w') as f:
            json.dump(data, f)

    def write_chromosome_data(self, chromosome, data):
        # contains run data for all chromosomes
        file_name = f"{''.join(map(str, chromosome))}.json"
        self.write(file_name, data)

    def write_final(self, chromosome, data, settings):
        # contains run data for final constraint
        file_name = f"{''.join(map(str, chromosome))}_BEST_{settings['algorithm']}" \
                    f"_{settings['setting']}_{settings['time_limit']}_{settings['domain']}.json"
        self.write(file_name, data)

    def write_genetic_stats(self, data):
        # contains: population for each round, fitness for each round
        file_name = "genetic_run_statistics.json"
        self.write(file_name, data)