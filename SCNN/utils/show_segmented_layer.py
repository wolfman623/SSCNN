import numpy as np
import skimage
import time
from matplotlib import pyplot as plt
from skimage.morphology import closing, square
from utils.evaluation import pixel_level_evaluate


def show_result(img_ori, labeled_image, predict_labels, num_classes, data_path):
    del(predict_labels[0])
    for layer in range(num_classes):
        img_result = img_ori.copy()
        area_this_layer = [i+1 for i, v in enumerate(predict_labels) if v == layer]
        position_tmp = 0
        for area_index in area_this_layer:
            position_tmp = position_tmp + (labeled_image == area_index)
        position_tmp = closing(position_tmp, square(2))
        GT = skimage.io.imread(data_path + "/GT%d.bmp" % (layer + 1))
        GT_gray = skimage.color.rgb2gray(GT)
        thresh = skimage.filters.threshold_otsu(GT_gray)
        GT_bw = np.logical_not(GT_gray > thresh)
        plt.figure()
        plt.imshow(GT_bw)
        plt.figure()
        plt.imshow(position_tmp)
        presision, recall, Dice = pixel_level_evaluate(position_tmp, GT_bw)
        print("Layer %d: precision is : %f, recall is : %f, Dice is : %f" % (layer+1, presision, recall, Dice))
        position_tmp = np.logical_not(position_tmp)
        img_result[position_tmp, :] = [255, 255, 255]
        plt.figure()
        plt.imshow(img_result)
        plt.imsave(data_path + "/AGWT_result%d.bmp" % (layer + 1), img_result)
    return None

def show_result_no_GT(img_ori, labeled_image, predict_labels, num_classes, data_path):
    start = time.time()
    predict_labels[0] = num_classes
    [img_height, img_width, img_chanel] = tuple(img_ori.shape)
    img_result = np.ones((img_height,img_width))*num_classes
    for height_ind in range(img_height):
        for width_ind in range(img_width):
            start = time.time()
            label_tmp = labeled_image[height_ind, width_ind]
            time_point1 = time.time()
            print(time_point1-start)
            img_result[height_ind, width_ind] = predict_labels[label_tmp]
        time_point2 = time.time()
        print(time_point2-time_point1)
    for layer_ind in range(num_classes):
        layer_img = img_ori.copy()
        layer_tmp = img_result==(layer_ind)
        layer_tmp = closing(layer_tmp, square(2))
        layer_tmp = np.logical_not(layer_tmp)
        layer_img[layer_tmp, :] = [255, 255, 255]
        plt.figure()
        plt.imshow(layer_img)
        plt.imsave(data_path + "/AGWT_result%d.bmp" % (layer_ind + 1), layer_img)

    # del (predict_labels[0])
    # for layer in range(num_classes):
    #     img_result = img_ori.copy()
    #     area_this_layer = [i+1 for i, v in enumerate(predict_labels) if v == layer]
    #     position_tmp = 0
    #     for area_index in area_this_layer:
    #         position_tmp = position_tmp + (labeled_image == area_index)
    #     position_tmp = closing(position_tmp, square(2))
    #     position_tmp = np.logical_not(position_tmp)
    #     img_result[position_tmp, :] = [255, 255, 255]
    #     plt.figure()
    #     plt.imshow(img_result)
    #     plt.imsave(data_path + "/AGWT_result%d.bmp" % (layer + 1), img_result)
    #     time_point = time.time()
    #     print(time_point-start)
    return None