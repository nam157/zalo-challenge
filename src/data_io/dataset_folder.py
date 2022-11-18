# -*- coding: utf-8 -*-
# @Time : 20-6-4 下午4:04
# @Author : zhuying
# @Company : Minivision
# @File : dataset_folder.py
# @Software : PyCharm

import cv2
import numpy as np
import torch
from torchvision import datasets


def opencv_loader(path):
    img = cv2.imread(path)
    return img


# class DatasetFolderFT(datasets.ImageFolder):
#     def __init__(self, root, transform=None, target_transform=None,
#                  ft_width=10, ft_height=10, loader=opencv_loader):
#         super(DatasetFolderFT, self).__init__(root, transform, target_transform, loader)
#         self.root = root
#         self.ft_width = ft_width
#         self.ft_height = ft_height

#     def __getitem__(self, index):
#         path, target = self.samples[index]
#         sample = self.loader(path)
#         # generate the FT picture of the sample
#         ft_sample = generate_FT(sample)
#         if sample is None:
#             print('image is None --> ', path)
#         if ft_sample is None:
#             print('FT image is None -->', path)
#         assert sample is not None

#         ft_sample = cv2.resize(ft_sample, (self.ft_width, self.ft_height))
#         ft_sample = torch.from_numpy(ft_sample).float()
#         ft_sample = torch.unsqueeze(ft_sample, 0)

#         if self.transform is not None:
#             try:
#                 sample = self.transform(sample)
#             except Exception as err:
#                 print('Error Occured: %s' % err, path)
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#         return sample, ft_sample, target


# def generate_FT(image):
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     f = np.fft.fft2(image)
#     fshift = np.fft.fftshift(f)
#     fimg = np.log(np.abs(fshift)+1)
#     maxx = -1
#     minn = 100000
#     for i in range(len(fimg)):
#         if maxx < max(fimg[i]):
#             maxx = max(fimg[i])
#         if minn > min(fimg[i]):
#             minn = min(fimg[i])
#     fimg = (fimg - minn+1) / (maxx - minn+1)
#     return fimg

import cv2
import numpy as np
import torch
from PIL import Image


def generate_FT(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    fimg = np.log(np.abs(fshift) + 1)
    maxx = -1
    minn = 100000
    for i in range(len(fimg)):
        if maxx < max(fimg[i]):
            maxx = max(fimg[i])
        if minn > min(fimg[i]):
            minn = min(fimg[i])
    fimg = (fimg - minn + 1) / (maxx - minn + 1)
    return fimg


def get_kernel(height, width):
    kernel_size = ((height + 15) // 16, (width + 15) // 16)
    return kernel_size


class Dataset:
    def __init__(self, label_list, transforms, ft_width, ft_height):
        self.file_list, self.label = self.get_file_list(label_list)
        self.transforms = transforms
        self.ft_width = ft_width
        self.ft_height = ft_height

    def __getitem__(self, index):

        img = cv2.imread(self.file_list[index])
        ft_sample = generate_FT(img)
        ft_sample = cv2.resize(ft_sample, (self.ft_width, self.ft_height))
        ft_sample = torch.from_numpy(ft_sample).float()
        ft_sample = torch.unsqueeze(ft_sample, 0)

        if self.transforms:
            img = self.transforms(img)

        label = self.label[index]

        return img, ft_sample, label

    def __len__(self):
        return len(self.file_list)

    def get_file_list(self, label_lists):
        file_list = []
        label_list = []
        for file in label_lists:  # open(label_lists, "r"):
            file_info = file.strip("\n").split(" ")
            file_name = file_info[0]
            label = file_info[1]
            file_list.append(file_name)
            label_list.append(int(label))
        return file_list, label_list
