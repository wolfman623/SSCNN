import numpy as np


def area_level_evaluate(predict_labels, truth_labels):
    area_num = len(truth_labels)
    positive = 0
    for ii in range(area_num):
        if truth_labels[ii]==predict_labels[ii]:
            positive += 1
    acc = positive / area_num
    return acc


def pixel_level_evaluate(segmented_result, groung_truth):
    true_positive = np.where(np.logical_and(segmented_result == 1, groung_truth == 1))
    segmented = np.where(segmented_result == 1)
    truth = np.where(groung_truth == 1)
    presision = true_positive[0].size/segmented[0].size
    recall = true_positive[0].size/truth[0].size
    Dice = 2*presision*recall/(presision+recall)
    return presision, recall, Dice