from numpy import ndarray


class NormalizationInfo:

    def __init__(self, mean: float, variance: float):
        self.mean = mean
        self.variance = variance


class Normalizer:

    def __init__(self, data: ndarray):
        self.normalization_info_list = []
        means = data.mean(axis=0)
        variances = data.var(axis=0)
        for i in range(0, len(means)):
            self.normalization_info_list.append(NormalizationInfo(means[i], variances[i]))

    def normalize_vector(self, vector: []) -> []:
        result = []
        for i in range(0, len(vector)):
            normalized_value = (vector[i] - self.normalization_info_list[i].mean) / \
                               self.normalization_info_list[i].variance
            result.append(normalized_value)
        return result
