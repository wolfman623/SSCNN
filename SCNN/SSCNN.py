import os
import torch
import numpy as np
import skimage
from skimage import io
import time
from collections import Counter
import matplotlib.image as mpimg
from utils.config_parser import parse_yaml_config
from torch.utils.data import DataLoader
from torch import nn
from network.shallowcnn import ShallowCNN
from sampler.select_points import SelectPoints
from sampler.data_prepare import DataPrepare
from utils.evaluation import area_level_evaluate
from utils.show_segmented_layer import show_result, show_result_no_GT


def train(config):
    config = parse_yaml_config(config)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.system.gpu_devices)
    log_dir = config.system.log_dir + "\\" + config.dataset.map_name + "\\"
    data_path = config.dataset.data_path + "\\" + config.dataset.map_name + "\\"
    ori_image = data_path + config.dataset.map_name + ".bmp"
    area_image = data_path + config.dataset.map_name + "_" + config.dataset.area_image + ".bmp"
    point_path = log_dir
    ground_truth_path = data_path
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # load the GT images
    img_area = mpimg.imread(area_image)
    img_ori = mpimg.imread(ori_image)
    if config.dataset.color_space == 'Lab':
        img_ori = skimage.color.rgb2lab(img_ori)
    elif config.dataset.color_space == 'RGB':
        pass
    else:
        raise ValueError


    if config.dataset.Normalization == False:
        img = img_ori
    elif config.dataset.Normalization == True: # Normalization of the image, in each channel
        img = np.zeros(img_ori.shape)
        for chanel in range(3):
            mu = np.average(img_ori[:, :, chanel])
            std = np.std(img_ori[:, :, chanel])
            img[:, :, chanel] = (img_ori[:, :, chanel] - mu) / std
    else:
        raise ValueError



    # convert the boundary map into connective region labeled map
    boundary = np.logical_not(np.logical_and(img_area[:,:,0]==255, img_area[:,:,1]==0, img_area[:,:,2]==0))
    labeled_image = skimage.measure.label(boundary, connectivity=2)
    area_num = np.max(labeled_image)

    # give each region a label
    GT_label_image = np.ones(labeled_image.shape)*(config.network.num_classes-1)
    for GT_num in range(config.network.num_classes-1):
        GT = skimage.io.imread(data_path +"/GT%d.bmp" % (GT_num+1))
        GT_gray = skimage.color.rgb2gray(GT)
        thresh = skimage.filters.threshold_otsu(GT_gray)
        GT_bw = np.logical_not(GT_gray > thresh)
        GT_label_image[GT_bw] = GT_num

    # store all the superpixels in all_area
    all_area = []
    for area_index in range(area_num):
        all_area.append([])
        area = labeled_image==area_index
        area_point_label = GT_label_image[area]
        area_point_coord = img[area,:]
        area_label = Counter(area_point_label).most_common(1)
        all_area[area_index].append(area_point_coord)
        all_area[area_index].append(area_label[0][0])

    Train_Points = SelectPoints(img_area = img_area,
                                labeled_image=labeled_image,
                                all_area=all_area,
                                ground_truth_path=ground_truth_path,
                                num_classes=config.network.num_classes,
                                point_path=point_path,
                                save_selected_points=True)
    if config.dataset.train.points_mouse_selected:
        selected_train_points = Train_Points.mouse_select()
    else:
        selected_train_points = Train_Points.load_saved_points()

    # create the model and train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ShallowCNN(num_classes=config.network.num_classes)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=config.network.learning_rate,
                                momentum=config.network.momentum)
    reduce_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.network.epochs, eta_min=0)
    # reduce_lr = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)

    # load training dataset
    #TODO: package the data loading in a function
    for epoch in range(config.network.epochs):
        print('epoch ', epoch)
        prepared_data = DataPrepare(samples_per_class=config.dataset.train.samples_per_class,
                                    num_classes=config.network.num_classes,
                                    sequence_len=config.dataset.sequence_len,
                                    area_num=area_num)
        dataset = prepared_data.train_data(selected_train_points=selected_train_points)
        train_data_loader = DataLoader(dataset=dataset,
                                       batch_size=config.dataset.train.batch_size,
                                       num_workers=0,
                                       pin_memory=True,
                                       shuffle=config.dataset.train.shuffle,
                                       drop_last=False)
        model.train(train_data_loader=train_data_loader,
                    device=device,
                    optimizer=optimizer,
                    criterion=criterion)
        reduce_lr.step()
        torch.save(model.state_dict(), os.path.join(log_dir, "model.dat"))
    print("training complete")
    return 0


def reload_model(config="config.yaml", strict=True):
    """
    Building a network from configs and loading weights from pretrained model
    :param config:
    :param strict:
    :return:
    """
    log_dir = config.system.log_dir + "\\" + config.dataset.map_name + "\\"
    if isinstance(config, (str,)):
        config = parse_yaml_config(config)
    model = ShallowCNN(**config.network._asdict())
    if not os.path.exists(os.path.join(log_dir, "model.dat")):
        raise ValueError("There's no model.dat in specific path. Please train a model first.")
    model.load_state_dict(torch.load(log_dir + "\model.dat"), strict=strict)
    return model


def segment(config):
    # start = time.time()
    config = parse_yaml_config(config)
    model = reload_model(config)
    log_dir = config.system.log_dir + "\\" + config.dataset.map_name + "\\"
    data_path = config.dataset.data_path + "\\" + config.dataset.map_name + "\\"
    ori_image = data_path + config.dataset.map_name + ".bmp"
    area_image = data_path + config.dataset.map_name + "_" + config.dataset.area_image + ".bmp"

    # test_data = load_mat_data('E:/Current research/SSCNN/dataset/map1/testImages_30_line.mat',key='testImages')
    # test_labels = load_mat_data('E:/Current research/SSCNN/dataset/map1/testLabels_30_line.mat',key='testLabels')
    # test_data = test_data.transpose((2, 0, 1))
    # test_data = test_data.reshape(test_data.shape[0], 1, test_data.shape[1], test_data.shape[2])
    # test_data = torch.FloatTensor(test_data)
    # dataset = LoadData(test_data, test_labels)

    # load the map
    img_area = mpimg.imread(area_image)
    img_ori = mpimg.imread(ori_image)
    if config.dataset.color_space == 'Lab':
        img_cov = skimage.color.rgb2lab(img_ori)
    elif config.dataset.color_space == 'RGB':
        img_cov = img_ori
    else:
        raise ValueError

    if config.dataset.Normalization == False:
        img = img_cov
    elif config.dataset.Normalization == True: # Normalization of the image, in each channel
        img = np.zeros(img_cov.shape)
        for chanel in range(3):
            mu = np.average(img_cov[:, :, chanel])
            std = np.std(img_cov[:, :, chanel])
            img[:, :, chanel] = (img_cov[:, :, chanel] - mu) / std
    else:
        raise ValueError

    # convert the boundary map into connective region labeled map
    boundary = np.logical_not(np.logical_and(img_area[:,:,0]==255, img_area[:,:,1]==0, img_area[:,:,2]==0))
    labeled_image = skimage.measure.label(boundary, connectivity=2)
    area_num = np.max(labeled_image) + 1
    [img_height,img_width,img_chanel] = tuple(img_ori.shape)
    # time_point1 = time.time()
    # print(time_point1-start)

    # give each region a label
    # GT_label_image = np.ones(labeled_image.shape)*(config.network.num_classes-1)
    # for GT_num in range(config.network.num_classes-1):
    #     GT = skimage.io.imread(data_path + "/GT%d.bmp" % (GT_num+1))
    #     GT_gray = skimage.color.rgb2gray(GT)
    #     thresh = skimage.filters.threshold_otsu(GT_gray)
    #     GT_bw = np.logical_not(GT_gray > thresh)
    #     GT_label_image[GT_bw] = GT_num

    # store all the superpixels in all_area
    # all_area = [None] * area_num
    all_area = []
    for _ in range(area_num):
        # all_area.append(np.empty([0,3]))
        all_area.append([])
    for height_ind in range(img_height):
        for width_ind in range(img_width):
            label_tmp = labeled_image[height_ind, width_ind]
            pixel_value = img[height_ind, width_ind,:]
            # pixel_value = pixel_value[:, np.newaxis]
            all_area[label_tmp].append(pixel_value)
            # all_area[label_tmp] = np.stack((all_area[label_tmp],pixel_value), axis=-1)
            # all_area[label_tmp] = np.vstack((all_area[label_tmp],pixel_value))
    for area_index in range(area_num):
        all_area[area_index] = np.array(all_area[area_index])
    # time_point2 = time.time()
    # print(time_point2-time_point1)

    # all_area = []
    # for area_index in range(area_num):
    #     start = time.time()
    #     all_area.append([])
    #     area = labeled_image==area_index
    #     # area_point_label = GT_label_image[area]
    #     area_point_coord = img[area,:]
    #     # area_label = Counter(area_point_label).most_common(1)
    #     all_area[area_index] = area_point_coord
    #     # all_area[area_index].append(area_point_coord)
    #     # all_area[area_index].append(0)
    #     end = time.time()
    #     print(end - start)
    predict_labels_all = []
    for loop in range(config.dataset.test.loop):
        prepared_data = DataPrepare(samples_per_class=config.dataset.train.samples_per_class,
                                    num_classes=config.network.num_classes,
                                    sequence_len=config.dataset.sequence_len,
                                    area_num=area_num)
        dataset = prepared_data.test_data(all_area=all_area)
        test_data_loader = DataLoader(dataset=dataset,
                                      batch_size=2000,
                                      num_workers=0,
                                      pin_memory=True,
                                      shuffle=False,
                                      drop_last=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        predict_labels = model.test(test_data_loader=test_data_loader,
                                                  device=device)
        # acc = area_level_evaluate(predict_labels=predict_labels,
        #                           truth_labels=truth_labels)
        predict_labels_all.append(predict_labels)
        # print(acc)
    # time_point3 = time.time()
    # print(time_point3-time_point2)
    predict_labels_all = np.transpose(predict_labels_all).tolist()  # 矩阵转list
    predict_labels =[Counter(predict_tmp).most_common(1)[0][0] for predict_tmp in predict_labels_all]
    show_result_no_GT(img_ori=img_ori,
                      labeled_image=labeled_image,
                      predict_labels=predict_labels,
                      num_classes=config.network.num_classes,
                      data_path=data_path)
    # time_point4 = time.time()
    # print(time_point4-time_point3)
    return