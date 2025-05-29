from statistics import mean, pstdev
from math import ceil, log10

class Normalizer:
    def __init__(self, data):
        self.data = data

    def min_max(self):
        Xmax = max(self.data)
        Xmin = min(self.data)

        diff = Xmax - Xmin

        return [((x - Xmin)/diff) for x in self.data]

    def z_score(self):
        Xmean = mean(self.data)
        Xstd_dev = pstdev(self.data)

        return [((x-Xmean)/Xstd_dev) for x in self.data]

    def decimal_scaling(self):
        abs_max = max([abs(x) for x in self.data])
        j = log10(abs_max)
        j = ceil(j)

        return [(x/(10**j)) for x in self.data]


# Test data
data = [123, 456, 789, -321, 0]

print(f"Raw data:\n{data}\n")

normalizer = Normalizer(data)

print(f"Min max normalization:\n{normalizer.min_max()}\n")
print(f"Z score normalization:\n{normalizer.z_score()}\n")
print(f"Decimal scaling:\n{normalizer.decimal_scaling()}\n")
