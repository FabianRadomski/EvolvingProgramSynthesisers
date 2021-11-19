from typing import List, get_type_hints
import copy

class Environment:
    """Abstract Environment class."""

    def __init__(self):
        self.program = None

    def distance(self, other: "Environment") -> float:
        """Returns the distance from this Environment to some other object."""
        raise NotImplementedError()

    def __deepcopy__(self, memdict={}):
        raise NotImplementedError()

    def correct(self, other: "Environment") -> bool:
        """Returns whether this state is the desired one given a desired output Environment."""
        raise NotImplementedError()


class RobotEnvironment(Environment):
    """Environment for the robot. A robot lives on a square matrix in which it needs to pick up a ball lying somewhere
    in that same matrix."""

    def __init__(self, size: int, rx: int, ry: int, bx: int, by: int, holding=False):
        """Creates new RobotEnvironment given a size, initial position of the robot (rx, ry), position of the ball
        (bx, by) and whether the robot is holding the ball."""
        super().__init__()

        self.size = size
        self.rx = rx
        self.ry = ry
        self.bx = bx
        self.by = by
        self.holding = holding

        assert (not holding or (rx == bx and ry == by))
        
    def __deepcopy__(self, memdict={}):
        return RobotEnvironment(self.size, self.rx, self.ry, self.bx, self.by, self.holding)

    def __str__(self):
        return "RobotEnvironment(Robot: (%s, %s), Bal: (%s, %s), Holding: %s)" % \
               (self.rx, self.ry, self.bx, self.by, self.holding)

    def distance(self, other: "RobotEnvironment") -> int:
        assert self.size == other.size

        def d(xy1: 'tuple[int, int]', xy2: 'tuple[int, int]'):
            return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

        # position robot and position ball
        pr = (self.rx, self.ry)
        pb = (self.bx, self.by)

        # position goal robot and position goal bal
        pgr = (other.rx, other.ry)
        pgb = (other.bx, other.by)

        if pr != pb and pb != pgb:
            return d(pr, pb) + d(pb, pgb) + d(pgb, pgr) + 2
        elif pr == pb and pb != pgb:
            return d(pr, pgb) + d(pgb, pgr) + 1
        else:
            return d(pr, pgr)

    def correct(self, other: "RobotEnvironment") -> bool:
        return (self.rx, self.ry, self.bx, self.by, self.holding) \
               == (other.rx, other.ry, other.bx, other.by, other.holding)


class StringEnvironment(Environment):
    """Environment for string manipulation."""
    def __init__(self, string: str, pos: int = 0):
        """Creates new StringEnvironment given an initial string and starting position of the pointer, 0 by default."""
        super().__init__()

        # Manipulating strings as a list of characters is more efficient.
        self.string_array = list(string)
        self.pos = pos
        
        assert 0 <= pos < len(string) or len(string) == 0

    def to_string(self) -> str:
        """Returns the string of this Environment. For efficiency strings are internally stored as an array instead of
        string, therefore this conversion method exists."""
        return "".join(self.string_array)

    def __deepcopy__(self, memdict={}):
        return StringEnvironment(self.to_string(), self.pos)
    
    @staticmethod
    def _levenshtein(str1, str2):
        m = len(str1)
        n = len(str2)
        d = [[i] for i in range(1, m + 1)]   # d matrix rows
        d.insert(0, list(range(0, n + 1)))   # d matrix columns
        for j in range(1, n + 1):
            for i in range(1, m + 1):
                if str1[i - 1] == str2[j - 1]:   # Python (string) is 0-based
                    substitutionCost = 0
                else:
                    substitutionCost = 1
                d[i].insert(j, min(d[i - 1][j] + 1,
                                d[i][j - 1] + 1,
                                d[i - 1][j - 1] + substitutionCost))
        return d[-1][-1]

    def distance(self, other: "StringEnvironment") -> int:
        return self._levenshtein("".join(self.string_array), "".join(other.string_array))

    def correct(self, other: "StringEnvironment") -> bool:
        return self.to_string() == other.to_string()

    def __str__(self):
        return "StringEnvironment(Pointer at {pos} in \"{string_array}\")".format(pos=self.pos, string_array=self.to_string())


class PixelEnvironment(Environment):
    def __init__(self, width, height, x, y, pixels=None):
        super().__init__()

        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.pixels = pixels
        if not pixels:
            self.pixels = [[False for _ in range(height)] for _ in range(width)]

        assert 0 <= x < width
        assert 0 <= y < height

    def __str__(self):
        return "PixelEnvironment((%s, %s), %s)" % (self.x, self.y, self.pixels)

    def __deepcopy__(self, memdict={}):
        return PixelEnvironment(self.width, self.height, self.x, self.y, list(map(list, self.pixels)))

    def _hamming_distance(self, matrix1: List[List[bool]], matrix2: List[List[bool]]) -> int:
        assert len(matrix1) == len(matrix2)
        assert len(matrix1[0]) == len(matrix2[0])
        element_list1 = [e for row in matrix1 for e in row]
        element_list2 = [e for row in matrix2 for e in row]
        diff = [abs(int(e1) - int(e2)) for (e1, e2) in zip(element_list1, element_list2)]
        return sum(diff)

    def correct(self, other: "PixelEnvironment") -> bool:
        return self._hamming_distance(self.pixels, other.pixels) == 0

    def distance(self, other: "PixelEnvironment") -> int:
        return self._hamming_distance(self.pixels, other.pixels)
