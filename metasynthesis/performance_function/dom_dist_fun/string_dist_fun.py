from typing import Callable, List

from common.environment.string_environment import StringEnvironment
from metasynthesis.performance_function.dom_dist_fun.domain_dist_fun import DomainDistFun


def f1(x: StringEnvironment, y: StringEnvironment) -> float:
    return abs(x.pos - y.pos)


def f2(x: StringEnvironment, y: StringEnvironment) -> float:
    return abs(len(x.string_array) - len(y.string_array))


def f3(x: StringEnvironment, y: StringEnvironment) -> float:
    return min(len(x.string_array), len(y.string_array)) - len(set(x.string_array) & set(y.string_array))


def count_upper(s):
    return sum(1 for c in s if c.isupper())


def count_lower(s):
    return sum(1 for c in s if c.islower())


def f4(x: StringEnvironment, y: StringEnvironment) -> float:
    return abs(count_upper(x.string_array) - count_upper(y.string_array))


def f5(x: StringEnvironment, y: StringEnvironment) -> float:
    return abs(count_lower(x.string_array) - count_lower(y.string_array))


class StringDistFun(DomainDistFun):

    @staticmethod
    def fun_to_string() -> dict:
        return {f1: 'd_p', f2: 'd_len', f3: 'm', f4: 'd_U', f5: 'd_L'}

    @staticmethod
    def partial_dist_funs() -> List[Callable[[StringEnvironment, StringEnvironment], float]]:
        return [f1, f2, f3, f4, f5]


if __name__ == '__main__':
    s1 = StringEnvironment(list('nikos123'), 0)
    s2 = StringEnvironment(list('nik'), 0)
    print(f3(s1, s2))
