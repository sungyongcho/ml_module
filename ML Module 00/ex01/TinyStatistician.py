import numpy as np
import math


class TinyStatistician(object):

    def __init__(self):
        pass

    def mean(self, x):
        sum = 0
        if x == []:
            return None
        for item in x:
            sum += item
        return float(sum / len(x))

    def median(self, x):
        if not x:
            return None
        sorted_x = sorted(x)
        n = len(x)
        if n % 2 == 0:
            return (sorted_x[n // 2 - 1] + sorted_x[n // 2]) / 2
        else:
            return sorted_x[n // 2]

    def quartile(self, x):
        if not x:
            return None
        sorted_x = sorted(x)
        n = len(x)
        if n % 2 == 0:
            return [self.median(sorted_x[:n // 2]), self.median(sorted_x[n // 2:])]
        else:
            return [self.median(sorted_x[:n // 2]), self.median(sorted_x[n // 2 + 1:])]

    def percentile(self, x, p):
        if not x:
            return None
        sorted_x = sorted(x)
        n = len(x)
        if n == 1:
            return sorted_x[0]
        k = (n - 1) * (p / 100)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_x[int(k)]
        d0 = sorted_x[int(f)] * (c - k)
        d1 = sorted_x[int(c)] * (k - f)
        return d0 + d1

    def var(self, x):
        if not isinstance(x, list) or len(x) == 0:
            return None
        mean = self.mean(x)
        suqared_diff_sum = 0.0
        for num in x:
            suqared_diff_sum += (num - mean) ** 2
        #suqared_diff_sum = np.sum((num - mean) ** 2)
        return suqared_diff_sum / (len(x) - 1)

    def std(self, x):
        return math.sqrt(self.var(x))


def ex2(t):
    lst = [14, 17, 10, 14, 18, 20, 13]
    print(t.mean(lst))
    print(t.median(lst))  # 14
    print(t.quartile(lst))  # 13, 18
    print(t.var(lst))
    print(t.std(lst))


def ex3(t):
    lst2 = [177, 180, 175, 182, 190, 169, 185, 191, 193]
    print(t.mean(lst2))
    print(t.median(lst2))  # 182
    print(t.quartile(lst2))  # 177 190
    print(t.var(lst2))
    print(t.std(lst2))


def ex4():
    a = [1, 42, 300, 10, 59]
    print(TinyStatistician().mean(a))  # Output: 82.4
    print(TinyStatistician().median(a))  # Output: 42.0
    print(TinyStatistician().quartile(a))  # Output: [10.0, 59.0]
    print(TinyStatistician().var(a))  # Output: 15349.3
    print(TinyStatistician().std(a))  # Output: 123.89229193133849


def ex5():
    a = [1, 42, 300, 10, 59]
    print(TinyStatistician().percentile(a, 10))  # Output: 4.6
    print(TinyStatistician().percentile(a, 15))  # Output: 6.4
    print(TinyStatistician().percentile(a, 20))  # Output: 8.2


if __name__ == "__main__":
    a = [1, 42, 300, 10, 59]
    print(TinyStatistician().mean(a))
    # Output:
    # 82.4
    print(TinyStatistician().median(a))
    # # Output:
    # 42.0
    print(TinyStatistician().quartile(a))
    # # Output:
    # [10.0, 59.0]
    print(TinyStatistician().percentile(a, 10))
    # # # Output:
    # # 4.6
    print(TinyStatistician().percentile(a, 15))
    # # # Output:
    # # 6.4
    print(TinyStatistician().percentile(a, 20))
    # # # Output:
    # # 8.2
    print(TinyStatistician().var(a))
    # # Output:
    # 15349.3
    print(TinyStatistician().std(a))
    # # Output:
    # 123.89229193133849
    print("=============other tests==============")
    t = TinyStatistician()
    print("\t# ex2")
    ex2(t)
    print("\t# ex2")
    print("\t# ex3")
    ex3(t)
    print("\t# ex3")
    print("\t# ex4")
    ex4()
    print("\t# ex4")
    print("\t# ex5")
    ex5()
    print("\t# ex5")
