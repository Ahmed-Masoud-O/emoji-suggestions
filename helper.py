import numpy as np
import math


def log2(x):
    return math.log(x) / math.log(2)


def f_measure(assignments, segments_array):
    f_measures = list()
    unique, counts = np.unique(segments_array, return_counts=True)
    segment_array_count_dict = dict(np.asarray((unique, counts)).T)
    precision_dict = {}
    count = 0
    for x in range(len(assignments)):
        if assignments[x] == segments_array[x]:
            count += 1
            if segments_array[x] in precision_dict:
                precision_dict[segments_array[x]] = precision_dict[segments_array[x]] + 1
            else:
                precision_dict[segments_array[x]] = 1
    key = max(precision_dict, key=precision_dict.get)
    value = precision_dict[max(precision_dict, key=precision_dict.get)]
    precision = value / count
    recall = value / int(segment_array_count_dict[key])
    f_measures.append(2 * precision * recall / (precision + recall))
    return np.asarray(f_measures).mean(), precision, recall


def conditional_entropy(assignments, segments_array):
    entropy_list = list()
    precision_dict = {}
    count = 0
    entropy = 0
    for x in range(len(assignments)):
        if assignments[x] == segments_array[x]:
            count += 1
            if segments_array[x] in precision_dict:
                precision_dict[segments_array[x]] = precision_dict[segments_array[x]] + 1
            else:
                precision_dict[segments_array[x]] = 1
    for key, value in precision_dict.items():
        entropy += -value / count * log2(value / count)
    entropy_list.append(entropy * count / len(assignments))
    return sum(entropy_list)
