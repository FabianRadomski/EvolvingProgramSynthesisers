
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

    res1 = Runner(dicts(0), "AS", setting, test_cases, 26.3, debug, store).run()
    res2 = Runner(dicts(0), "AS", setting, test_cases, 26.4, debug, store).run()
    res3 = Runner(dicts(0), "MH", setting, test_cases, 20.3, debug, store).run()
    res4 = Runner(dicts(0), "MH", setting, test_cases, 19.5, debug, store).run()
    print(f"1.AStar 26.3 Solved {str(res1[0])}, time: {str(res1[1])}\n")
    print(f"2.AStar 26.4 Solved {str(res2[0])}, time: {str(res2[1])}\n")
    print(f"3.MH 20.3 Solved {str(res3[0])}, time: {str(res3[1])}\n")
    print(f"4.MH 19.5 Solved {str(res4[0])}, time: {str(res4[1])}\n")