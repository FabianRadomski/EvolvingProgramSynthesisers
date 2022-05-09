import itertools
from common.program_synthesis.dsl import DomainSpecificLanguage, StandardDomainSpecificLanguage
from common.tokens.control_tokens import If, LoopWhile
from common.tokens.string_tokens import *


# Generates all permutations of elements in a set where maxLength > len(per) > 1
def generatePermutations(dsl, maxLength) -> list:
    if (maxLength <= 0):
        return []
    ret = list(filter(lambda p: p[-1] in dsl.get_trans_tokens(p[:-1]), itertools.permutations(dsl.get_trans_tokens(), maxLength))) + generatePermutations(dsl, maxLength - 1)
    return ret


# Composes tokens into Invented tokens
# Returns a list of Invented tokens
def inventTokens(dsl, maxLength) -> list:
    perms = generatePermutations(dsl, maxLength)
    out = []

    # convert these into "invented tokens"
    for p in perms:
        if len(p) > 1:
            p = list(map(lambda x: x, p))
            out.append(InventedToken(p))
        else:
            out.append(p[0])

    return out


# Composes tokens into more elaborate Invented tokens
# Also generates If and While tokens
def invent2(dsl: DomainSpecificLanguage, maxLength) -> list:
    # Normal invention step
    out = inventTokens(dsl, maxLength)

    # Generating if statements
    if_list = []
    conditions = dsl.get_bool_tokens()
    bodies = inventTokens(dsl, max(1, int(maxLength / 2)))  # TODO Arbitrary length!!
    for c in conditions:
        for lb in bodies:
            for rb in bodies:
                if_list.append(If(c,  (lb if isinstance(lb, list) else [lb]),  (rb if isinstance(rb, list) else [rb])))
    out = out + if_list

    # Generating recurse statements
    # recurse_list = []
    # conditions = boolTokenSet
    # conditions
    # bodies = inventTokens(tokenSet, max(1, int(maxLength / 2)))  # TODO Arbitrary length!!
    # for c in conditions:
    #     for lb in bodies:
    #         for rb in bodies:
    #             recurse_list.append(Recurse(c(), [lb], [rb]))
    #         recurse_list.append(Recurse(c(), [lb], []))
    #         recurse_list.append(Recurse(c(), [], [lb]))

    # for lb in bodies:
    #     for rb in bodies:
    #         recurse_list.append(Recurse(None, [lb], [rb]))
    #     recurse_list.append(Recurse(None, [lb], []))
    #     recurse_list.append(Recurse(None, [], [lb]))
    # out = out + recurse_list
    loop_list = []
    bodies = inventTokens(dsl, max(1, int(maxLength / 2)))  # TODO Arbitrary length!!
    for c in conditions:
        for lb in bodies:
            loop_list.append(LoopWhile(c, (lb if isinstance(lb, list) else [lb])))
    out = out + loop_list
    return out


if __name__ == "__main__":
    dsl = DomainSpecificLanguage('robot',
                                 [AtStart()],
                                 [MoveLeft(), MoveRight()],
                                 True,
                                 lambda x: [MoveLeft(), MakeLowercase()])
    out = invent2(dsl, 2)
    print(len(out))
    for t in out:
        print(t)
        if (not isinstance(t, Token)):
            print(t)
