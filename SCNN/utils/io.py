# coding: utf-8
# By Deserts
import numpy as np
# import nibabel as nib
import os
import csv
import scipy.io as io


def get_image_reader(file_format="nii"):
    MAP = {
        "nii": load_nii_data,
        "mat": load_mat_data
    }
    return MAP[file_format]


def get_image_writer(file_format="nii"):
    MAP = {
        "nii": save_nii_data_simple,
        "mat": save_mat_data
    }
    return MAP[file_format]


def get_file_list(csv_file_path):
    file_path_list = []
    with open(csv_file_path, 'r') as fp:
        reader = csv.reader(fp)
        for row in reader:
            file_path_list.append(row)
    return file_path_list


def load_mat_data(path, key="scene"):
    data = io.loadmat(path)
    scene = data[key]
    return scene


def save_mat_data(data, path, key="scene", do_compression=True):
    mat_dict = {key: data}
    if not path.endswith(".mat"):
        path = path + ".mat"
    io.savemat(path, mat_dict, do_compression=do_compression)


def load_nii_data(path):
    """
    Loads the nii image specified by path.
    The image can be 2D, but will be returned as 3D, with dimensions =[x, y, 1]
    It can also be 4D, of shape [x,y,z,1], and will be returned as 3D.
    If it's 4D with 4th dimension > 1, assertion will be raised.
    :param path: file path for the image
    :return: a 3D np array.
    """
    proxy = nib.load(path)
    image = proxy.get_data()
    proxy.uncache()

    if len(image.shape) == 2:
        # 2D image could have been given.
        image = np.expand_dims(image, axis=2)
    elif len(image.shape) > 3:
        # 4D volumes could have been given. Often 3Ds are stored as 4Ds with 4th dim == 1.
        assert image.shape[3] <= 1
        image = image[:, :, :, 0]
    return image


def save_nii_data_simple(data, target_path):
    """
    Save nii image to disk.
    :param data: 3D np array
    :param target_path: file path where to save.
    :return:
    """

    image = nib.Nifti1Image(data, np.eye(4))
    image.set_data_dtype(data.dtype)

    target_path = os.path.abspath(target_path)
    if not target_path.endswith(".nii.gz"):
        target_path = target_path + ".nii.gz"
    return nib.save(image, target_path)


def save_nii_data(data, target_path, original_path, dtype=np.dtype(np.float32)):
    """
    Save nii image to disk.
    :param data: 3D np array
    :param target_path: file path where to save.
    :param original_path: original image, where to copy the header over to the target image.
    :param dtype:
    :return:
    """
    proxy_origin = nib.load(original_path)
    header_origin = proxy_origin.header
    affine_origin = proxy_origin.affine
    proxy_origin.uncache()

    image = nib.Nifti1Image(data, affine_origin)
    image.set_data_dtype(dtype)

    zooms = list(header_origin.get_zooms()[:len(data.shape)])
    if len(zooms) < len(data.shape):
        zooms = zooms + [1.0] * (len(data.shape) - len(zooms))
    image.header.set_zooms(zooms)

    target_path = os.path.abspath(target_path)
    if not target_path.endswith(".nii.gz"):
        target_path = target_path + ".nii.gz"
    return nib.save(image, target_path)


if __name__ == '__main__':
    pass
