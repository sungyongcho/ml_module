import numpy


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
        if x == []:
            return None
        if len(x) == 1:
            return float(x[0])
        tmp = x[:]
        tmp.sort()
        if (len(tmp) % 2 == 0):
            return float((tmp[int(len(tmp) / 2)] + tmp[int(len(tmp) / 2) + 1])
                         / 2)
        else:
            return float(tmp[int(len(tmp) / 2)])

    def quartile(self, x):
        tmp = x[:]
        tmp.sort()
        median_val = self.median(x)
        # print("tmp", [i for i in tmp if i <= median_val],
        #       [i for i in tmp if i >= median_val])
        if (len(x) % 2 == 0):
            return [self.median(tmp[0:int(len(tmp) / 2)]),
                    self.median(tmp[int(len(tmp) / 2):len(tmp)])]
        else:
            return [self.median(tmp[0:int(len(tmp) / 2) + 1]),
                    self.median(tmp[int(len(tmp) / 2): len(tmp)])]

    def var(self, x):
        mean = self.mean(x)
        sum = 0
        for item in x:
            sum += (item - mean) ** 2
        return sum / len(x)

    def std(self, x):
        return self.var(x)**(1/2)


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
    # TinyStatistician().percentile(a, 10)
    # # # Output:
    # # 4.6
    # TinyStatistician().percentile(a, 15)
    # # # Output:
    # # 6.4
    # TinyStatistician().percentile(a, 20)
    # # # Output:
    # # 8.2
    print(TinyStatistician().var(a))
    # # Output:
    # 15349.3
    print(TinyStatistician().std(a))
