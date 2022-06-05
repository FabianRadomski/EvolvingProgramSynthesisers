import os
from multiprocessing import Pool
from multiprocessing.managers import BaseManager

from random import randint

def example_function(a):
    new_numbers = [randint(1, a) for i in range(0, 50)]

    with Pool(processes=os.cpu_count()-1) as pool:
        results = pool.map(str, new_numbers)

    results = []
    for nn in new_numbers:
        results.append(str(nn))

    return results


if __name__ == '__main__':

    numbers = [randint(1, 50) for i in range(0, 50)]

    manager = BaseManager(address=('', 50000), authkey=b'abc')
    server = manager.get_server()
    server.serve_forever()


    with Pool(processes=os.cpu_count()) as pool:
        results = pool.map(example_function, numbers)

    # results = []
    # for n in numbers:
    #     results.append(example_function(n))

    print("Final results:", results)

