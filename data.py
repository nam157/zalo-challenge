import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from transform import transformer
from utils import convert_bbox


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


class ContestDataset(Dataset):
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
        for file in label_lists:
            file_info = file.strip("\n").split(" ")
            file_name = file_info[0]
            label = file_info[1]
            file_list.append(file_name)
            label_list.append(int(label))
        return file_list, label_list


class CelebADataset(Dataset):
    def __init__(self, base_dir, label_path, transforms, ft_width, ft_height):
        self.base_dir = base_dir
        self.label_path = label_path
        self._load_data()
        self.transforms = transforms
        self.ft_width = ft_width
        self.ft_height = ft_height
    
    def _load_data(self):
        with open(os.path.join(self.base_dir, self.label_path), 'r') as f:
            data = f.readlines()
            self.data = [each.replace('\n', '') for each in data]

    def __getitem__(self, index):
        img_path, liveness_label = self.data[index].split()
        img = cv2.imread(os.path.join(self.base_dir, img_path))
        h, w, _ = img.shape
        with open(os.path.join(self.base_dir, img_path)[:-4]+'_BB.txt', 'r') as f:
            bbox = f.readlines()
            bbox = [each.replace('\n', '') for each in bbox]
            bbox = bbox[0].split()[:-1]
            bbox = list(map(int, bbox))
            bbox = convert_bbox(bbox, w, h)
        
        img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]

        ft_sample = generate_FT(img)
        ft_sample = cv2.resize(ft_sample, (self.ft_width, self.ft_height))
        ft_sample = torch.from_numpy(ft_sample).float()
        ft_sample = torch.unsqueeze(ft_sample, 0)

        if self.transforms:
            img = self.transforms(img)

        return img, ft_sample, 1-int(liveness_label)

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    # labels = open(
    #     "/home/ai/challenge/darf-nam/zalo-challenge/datasets/img_crops/scale_1/file_label.txt",
    #     "r",
    # )
    # data_labels = labels.readlines()
    # label_train, label_val = train_test_split(
    #     data_labels, test_size=0.2, random_state=1
    # )
    # kernel_size = get_kernel(128, 128)
    # ft_h, ft_w = 2 * kernel_size[0], 2 * kernel_size[1]
    # data = ContestDataset(
    #     label_list=label_train,
    #     transforms=transformer["train"],
    #     ft_height=ft_h,
    #     ft_width=ft_w,
    # )
    # img,ft,label = data[0]
    # print(img.shape)
    # print(ft.shape)
    # print(label)

    kernel_size = get_kernel(128, 128)
    ft_h, ft_w = 2 * kernel_size[0], 2 * kernel_size[1]
    dataset = CelebADataset(
        '/opt/data/celeba_spoof/CelebA_Spoof',
        'metas/intra_test/test_label.txt',
        transformer['train'],
        ft_height=ft_h,
        ft_width=ft_w
    )