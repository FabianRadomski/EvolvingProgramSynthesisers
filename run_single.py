
from solver.runner.algorithms import dicts
from solver.runner.runner import Runner

if __name__ == "__main__":
    time_limit = 0.2
    debug = False
    store = False
    setting = "RO"
    algo = "MH"
    test_cases = "eval"

    store = False if test_cases == "param" else store

    mean1 = Runner(dicts(0), algo, setting, test_cases, time_limit, debug, store).run()

    print(f"Solved {str(mean1)}")