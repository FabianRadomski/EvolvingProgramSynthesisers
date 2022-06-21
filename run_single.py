from solver.runner.algorithms import dicts
from solver.runner.runner import Runner

if __name__ == "__main__":
    time_limit = 0.2
    debug = False
    store = False
    setting = "SO"
    test_cases = "eval"

    store = False if test_cases == "param" else store

    print(f"Running {setting} with {test_cases}")
    res1 = Runner(dicts(alg_sequence=[("Brute", 0.0515), ("Brute", 0.0515)]), "CS", setting, test_cases, 1000, debug, store).run()
    print(f"1. Solved {str(res1[0])}, time: {str(res1[1])}\n")
    res2 = Runner(dicts(alg_sequence=[("Brute", 0.0407), ("Brute", 0.0471)]), "CS", setting, test_cases, 1000, debug, store).run()
    print(f"2. Solved {str(res2[0])}, time: {str(res2[1])}\n")
    res3 = Runner(dicts(alg_sequence=[("Brute", 0.03896), ("AS", 0.05674)]), "CS", setting, test_cases, 1000, debug, store).run()
    print(f"3. Solved {str(res3[0])}, time: {str(res3[1])}\n")
    res4 = Runner(dicts(alg_sequence=[("Brute", 0.0529), ("AS", 0.0316)]), "CS", setting, test_cases, 1000, debug, store).run()
    print(f"4. Solved {str(res4[0])}, time: {str(res4[1])}\n")

    # time_limit = 0.2
    # debug = False
    # store = False
    # setting = "RO"
    # algo = "MH"
    # test_cases = "eval"
    #
    # store = False if test_cases == "param" else store
    #
    # res1 = Runner(dicts(0), "AS", setting, test_cases, 26.3, debug, store).run()
    # res2 = Runner(dicts(0), "AS", setting, test_cases, 26.4, debug, store).run()
    # res3 = Runner(dicts(0), "MH", setting, test_cases, 20.3, debug, store).run()
    # res4 = Runner(dicts(0), "MH", setting, test_cases, 19.5, debug, store).run()
    # print(f"1.AStar 26.3 Solved {str(res1[0])}, time: {str(res1[1])}\n")
    # print(f"2.AStar 26.4 Solved {str(res2[0])}, time: {str(res2[1])}\n")
    # print(f"3.MH 20.3 Solved {str(res3[0])}, time: {str(res3[1])}\n")
    # print(f"4.MH 19.5 Solved {str(res4[0])}, time: {str(res4[1])}\n")