import skimage
from matplotlib import pyplot as plt
from utils.io import load_mat_data, save_mat_data
from collections import Counter


class SelectPoints:
    def __init__(self, img_area, labeled_image, all_area, ground_truth_path, num_classes, point_path,
                save_selected_points=True):
        self.img_area = img_area
        self.labeled_image = labeled_image
        self.all_area = all_area
        self.ground_truth_path = ground_truth_path
        self.point_path = point_path
        self.num_classes = num_classes
        self.save_selected_points = save_selected_points

    def mouse_select(self):       # select the training data from the maps though using mouse clicking
        selected_train_points = []
        for layer_num in range(self.num_classes):
            selected_area_index = []
            single_layer_selected_train_points = []
            GT = skimage.io.imread(self.ground_truth_path + "/GT%d.bmp" % (layer_num + 1))
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(self.img_area)
            plt.subplot(1, 2, 2)
            plt.imshow(GT)
            pos = plt.ginput(-1, timeout=-1)
            save_mat_data(pos, self.ground_truth_path + "/selected_points/pos%d.mat" % (layer_num + 1), key='pos')
            for selec_point in pos:
                area_index = self.labeled_image[selec_point[1].astype(int)][selec_point[0].astype(int)]
                selected_area_index.append(area_index)
            selected_area_index = dict(Counter(selected_area_index))
            selected_area_index = list(selected_area_index)
            if 0 in selected_area_index:
                selected_area_index.remove(0)
            for area_index in selected_area_index:
                if  self.all_area[area_index][1] != layer_num:
                    continue
                single_layer_selected_train_points.extend( self.all_area[area_index][0])
            selected_train_points.append(single_layer_selected_train_points)
        return selected_train_points

    def load_saved_points(self):
        selected_train_points = []
        for layer_num in range(self.num_classes):
            selected_area_index = []
            single_layer_selected_train_points = []
            pos = load_mat_data(self.ground_truth_path + "/selected_points/pos%d.mat" % (layer_num + 1), key='pos')
            for selec_point in pos:
                area_index = self.labeled_image[selec_point[1].astype(int)][selec_point[0].astype(int)]
                selected_area_index.append(area_index)
            selected_area_index = dict(Counter(selected_area_index))
            selected_area_index = list(selected_area_index)
            if 0 in selected_area_index:
                selected_area_index.remove(0)
            for area_index in selected_area_index:
                if self.all_area[area_index][1] != layer_num:
                    continue
                single_layer_selected_train_points.extend(self.all_area[area_index][0])
            selected_train_points.append(single_layer_selected_train_points)
        return selected_train_points