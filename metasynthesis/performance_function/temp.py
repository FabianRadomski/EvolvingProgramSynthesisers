from math import sqrt
from random import randint

p = 0.6
h_max = 10


def pr_height_i_given_depth_d(i, d):
    if (i == 0 and d == 0) or (i + d > h_max):
        return 0
    elif i == 0 and d < h_max:
        return 1 - p
    elif i == 0 and d == h_max:
        return 1
    elif i > 0 and d == 0:
        rec = pr_height_i_given_depth_d(i - 1, 1)
        s = 0
        for k in range(i - 1):
            s += pr_height_i_given_depth_d(k, 1)
        return 2 * rec * s + rec ** 2
    elif i > 0 and d > 0:
        rec = pr_height_i_given_depth_d(i - 1, d + 1)
        s = 0
        for k in range(i - 1):
            s += pr_height_i_given_depth_d(k, d + 1)
        return p * (2 * rec * s + rec ** 2)


def pr_height_i(i):
    return pr_height_i_given_depth_d(i, 0)


def expected_height_random_tree():
    exp = 0
    for i in range(h_max + 1):
        exp += i * pr_height_i(i)
    return exp


def stand_deviation_of_height():
    e = 0
    for i in range(h_max + 1):
        e += (i ** 2) * pr_height_i(i)
    return sqrt(e - (expected_height_random_tree() ** 2))


# def pr_random_subtree_height_i(i):
#     pr = 0
#     for k in range(h_max - i + 1):
#         pr += pr_height_i_given_depth_d(i + k, k)
#     return pr

def pr_random_subtree_height_i(i):
    if i == 0:
        pr = 0
        for d in range(1, h_max + 1):
            for k in range(1, d + 1):
                pr += pr_height_i_given_depth_d(0, k) * ((h_max - k + 1) / (h_max + 1))
            # pr *= 1 / (h_max + 1)
        return pr
    else:
        pr = 0
        for k in range(h_max - i + 1):
            pr += pr_height_i_given_depth_d(i, k) / (h_max + 1)
        return pr


def expected_height_random_subtree():
    exp = 0
    for k in range(0, h_max + 1):
        exp += pr_random_subtree_height_i(k) * k
    return exp


def probability_height_i_AND_depth_j(i, j):
    return pr_height_i(i) * pr_height_i(j)


if __name__ == '__main__':
    for i in range(h_max + 1):
        print('Pr[h(T)=%s] = %s' % (i, pr_height_i(i)))
    print('--------')
    print('Expected height of random tree:', expected_height_random_tree())
    print('Standard deviation of the height:', stand_deviation_of_height())

    # s = 0
    # for k in range(h_max + 1):
    #     s += k * pr_random_subtree_height_i(k)
    # print(s)
    #
    # print(pr_random_subtree_height_i(0))
    print('HERE:', expected_height_random_subtree())
    total = 0
    correct_probabilities = {0: 0.643, 1: 0.118, 2: 0.0487, 3: 0.025, 4: 0.01678, 5: 0.01285, 6: 0.0127, 7: 0.0145,
                             8: 0.0208, 9: 0.0326, 10: 0.053}
    for i in range(h_max + 1):
        print('Pr[h(RandWalk(T))=%s] = %s, correct:%s' % (i, pr_random_subtree_height_i(i), correct_probabilities[i]))
        total += pr_random_subtree_height_i(i)
    print(total)
    print(pr_height_i_given_depth_d(0, 2))
    print('here', pr_height_i(2))
    print('here2', pr_height_i(1))
    for i in range(h_max + 1):
        print(pr_height_i_given_depth_d(i, 2) * (2 ** (i + 1)))
    # Pr[Height] at depth 2: [0.80275455 0.0398426  0.02557796 0.00885391 0.00442696 0.00393507
    #  0.00393507 0.0029513  0.10772258 0.         0.        ]
    print(probability_height_i_AND_depth_j(2, 2))
    print(pr_height_i(2) * 4 * (1 - p))
    print(pr_height_i(2))
