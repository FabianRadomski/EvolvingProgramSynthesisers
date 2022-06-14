from typing import Callable, Tuple, Dict, Iterable, List

import pandas
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from metasynthesis.language_constraints.results_parsing.Data_Reader import DataReader
from metasynthesis.language_constraints.results_parsing.Data_Compiler import *


def get_data(domains: List[str], functions: List[Tuple[str, Callable]]):
    writer = pd.ExcelWriter('file2.xlsx', engine='xlsxwriter')
    for sheet_name, function in functions:
        final = pd.DataFrame([], columns=["runtime",
                                          "domain name",
                                          "search_type",
                                          "complexity",
                                          "value"])
        for domain in domains:
            dr = DataReader(domain)
            data = dr.get_evaluation_data()
            norm, cons = function(data)
            data = []
            for key in norm:
                for complexity, value in norm[key]:
                    data.append((key, domain, "normal", complexity, value))
                for complexity, value in cons[key]:
                    data.append((key, domain, "constrained", complexity, value))
            data = pd.DataFrame(data,
                                columns=["runtime",
                                         "domain name",
                                         "search_type",
                                         "complexity",
                                         "value"])
            final = pd.concat([final, data])
        final.to_excel(writer, sheet_name=sheet_name)
    writer.save()




if __name__ == '__main__':
    domains = [
        "pixelBrutePE",
        "pixelBrutePG",
        "pixelBrutePO",
        "robotBruteRE",
        "robotBruteRG",
        "robotBruteRO"
    ]

    functions = [
        ("cost", cost_evaluation),
        ("runtime", runtime_evaluation)
    ]
    get_data(domains, functions)