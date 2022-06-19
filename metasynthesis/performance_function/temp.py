from math import sqrt

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



if __name__ == '__main__':
    for i in range(h_max + 1):
        print('Pr[h(T)=%s] = %s' % (i, pr_height_i(i)))
    print('--------')
    print('Expected height of random tree:', expected_height_random_tree())
    print('Standard deviation of the height:', stand_deviation_of_height())
