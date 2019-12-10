import random
import numpy as np
import torch
import math
from utils.loaddata import LoadData, LoadTestData


class DataPrepare:
    def __init__(self, samples_per_class, num_classes, sequence_len, area_num):
        self.samples_per_class = samples_per_class
        self.num_classes = num_classes
        self.sequence_len = sequence_len
        self.area_num = area_num

    def train_data(self,selected_train_points):
        train_data = []
        train_labels = []
        for ii in range(self.num_classes):
            for jj in range(self.samples_per_class):
                tmp = random.sample(selected_train_points[ii], self.sequence_len)
                train_data.append(tmp)
            train_labels.extend(np.ones(self.samples_per_class) * ii)
        # train_labels = torch.LongTensor(train_labels)
        # train_labels = torch.zeros(train_labels.shape[0], config.network.num_classes+1).scatter_(1, train_labels, 1)
        # train_labels = train_labels[:,1:]
        # train_labels = train_labels.long()
        train_data = np.array(train_data)
        train_labels = np.array(train_labels)
        train_labels = train_labels.reshape(train_labels.size, 1)
        train_data = train_data.reshape(train_data.shape[0], 1, train_data.shape[1], train_data.shape[2])
        train_data = torch.FloatTensor(train_data)
        dataset = LoadData(train_data, train_labels)
        return dataset

    def test_data(self, all_area):
        test_data = []
        # test_labels = []
        for area_index in range(self.area_num):
            area_length = all_area[area_index].shape[0]
            if area_length < self.sequence_len:
                repeat_time = math.ceil(self.sequence_len / area_length)
                area_tmp = np.repeat(all_area[area_index], repeat_time, axis=0)
            else:
                area_tmp = all_area[area_index]
            sequence_tmp = random.sample(list(area_tmp), self.sequence_len)
            test_data.append(sequence_tmp)
            # test_labels.append(all_area[area_index][1])
        test_data = np.array(test_data)
        test_data = test_data.reshape(test_data.shape[0], 1, test_data.shape[1],test_data.shape[2])
        test_data = torch.FloatTensor(test_data)
        # test_labels = np.array(test_labels)
        # test_labels = test_labels.reshape(test_labels.size, 1)
        # dataset = LoadData(test_data,test_labels)
        dataset = LoadTestData(test_data)
        return dataset