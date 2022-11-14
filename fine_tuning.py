import math
import os
from typing import Dict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import MultiFTNet
from data import Dataset
from transform import transformer
from utils import MiniFASNetV1, MiniFASNetV1SE, MiniFASNetV2, MiniFASNetV2SE

MODEL_MAPPING = {
    "MiniFASNetV1": MiniFASNetV1,
    "MiniFASNetV2": MiniFASNetV2,
    "MiniFASNetV1SE": MiniFASNetV1SE,
    "MiniFASNetV2SE": MiniFASNetV2SE,
}
SCALE = 2.7
HEIGHT = 80
WIDTH = 80
MODEL_TYPE = "MiniFASNetV2"


def parse_model_name(model_name):
    info = model_name.split("_")[0:-1]
    h_input, w_input = info[-1].split("x")
    model_type = model_name.split(".pth")[0].split("_")[-1]

    if info[0] == "org":
        scale = None
    else:
        scale = float(info[0])
    return int(h_input), int(w_input), model_type, scale


def get_kernel(height, width):
    kernel_size = ((height + 15) // 16, (width + 15) // 16)
    return kernel_size


class anti_spoofing:
    def __init__(
        self,
        label_path: str,
        params: Dict,
        model_path: str = None,
        pre_trained: bool = False,
    ) -> None:
        self.device = torch.device("cpu")
        if pre_trained:
            model_name = os.path.basename(model_path)
            h_input, w_input, model_type, _ = parse_model_name(model_name)
            self.kernel_size = get_kernel(
                h_input,
                w_input,
            )
            self.model = MODEL_MAPPING[model_type](img_channel=3,conv6_kernel=self.kernel_size).to(
                self.device
            )
            state_dict = torch.load(model_path, map_location=self.device)
            keys = iter(state_dict)
            first_layer_name = keys.__next__()
            if first_layer_name.find("module.") >= 0:
                from collections import OrderedDict

                new_state_dict = OrderedDict()
                for key, value in state_dict.items():
                    name_key = key[7:]
                    new_state_dict[name_key] = value
                self.model.load_state_dict(new_state_dict)
            else:
                self.model.load_state_dict(state_dict)
        else:
            self.kernel_size = get_kernel(80, 80)
            self.model = MODEL_MAPPING[MODEL_TYPE](conv6_kernel=self.kernel_size).to(self.device)
            print(self.model)
            # self.model = MultiFTNet(num_classes=2,img_channel=3,embedding_size=128,conv6_kernel=get_kernel(128, 128)).to(self.device)
        self.train_data_loader, self.val_data_loader = self._init_data(
            label_path, params
        )
        (
            self.optimizer,
            self.scheduler,
            self.cls_criterion,
            self.ft_criterion,
        ) = self._init_params()

    def _init_data(self, label_path: str, params: Dict):
        fh = open(label_path, "r")
        labels = fh.readlines()
        label_train, label_valid = train_test_split(
            labels, test_size=0.2, random_state=121
        )
        kernel_size = get_kernel(80, 80)
        ft_h, ft_w = 2 * kernel_size[0], 2 * kernel_size[1]
        data_train = Dataset(
            label_list=label_train,
            transforms=transformer["train"],
            ft_height=ft_h,
            ft_width=ft_w,
        )
        data_val = Dataset(
            label_list=label_valid,
            transforms=transformer["test"],
            ft_height=ft_h,
            ft_width=ft_w,
        )

        train_data_loader = DataLoader(data_train, **params)
        val_data_loader = DataLoader(data_val, **params)
        return train_data_loader, val_data_loader

    def _init_params(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10000, eta_min=1e-7
        )
        cls_criterion = CrossEntropyLoss()
        ft_criterion = MSELoss()

        return optimizer, scheduler, cls_criterion, ft_criterion

    def _train_one_epoch(self, device):
        # self.model.train()
        loss_sum = 0
        for img, feature_map, label in tqdm(self.train_data_loader, desc="Training"):
            img = img.to(device)
            label = label.to(device)
            feature_map = feature_map.to(device)
            print(img.shape,label,feature_map.shape)
            self.optimizer.zero_grad()
            embedding, feature_map = self.model(img)
            loss_cls = self.cls_criterion(embedding, label)
            loss_fea = self.ft_criterion(feature_map, feature_map)

            train_loss = 0.5 * loss_cls + 0.5 * loss_fea
            train_loss.backward()
            self.optimizer.step()

            loss_sum += train_loss
        return loss_sum / len(self.train_data_loader)

    def _valid_one_epoch(self, device):
        self.model.eval()
        loss_sum = 0
        for img, feature_map, label in tqdm(self.val_data_loader, desc="Validation"):
            with torch.no_grad():
                img = img.to(device)
                label = label.to(device)
                feature_map = feature_map.to(device)

                embedding, feature_map = self.model(img)
                loss_cls = self.cls_criterion(embedding, label)
                loss_fea = self.ft_criterion(feature_map, feature_map)

                valid_loss = 0.5 * loss_cls + 0.5 * loss_fea

                loss_sum += valid_loss
        return loss_sum / len(self.val_data_loader)

    def fit(self, epochs):
        
        for epoch in range(epochs):
            loss_train = self._train_one_epoch(device=self.device)
            print(f"Epochs: {epoch}/{epochs}--- Loss-Train: {loss_train}")
            if epoch % 2 == 0:
                loss_val = self._valid_one_epoch(device=self.device)
                print(
                    f"Epochs: {epoch}/{epochs}--- Loss-Train: {loss_train} ----- Loss-Valid: {loss_val}"
                )
                save_dir = "ckpt/anti_spoofing/"
                os.makedirs(save_dir, exist_ok=True)
                torch.save(
                    self.model.state_dict(),
                    save_dir + f"{SCALE}_{HEIGHT}x{WIDTH}_{MODEL_TYPE}.pth",
                )
        self.scheduler.step()


if __name__ == "__main__":
    params = {"batch_size": 4, "shuffle": True, "num_workers": 0, "pin_memory": True}
    deep_fake = anti_spoofing(
        label_path="G:/zalo_challenge/liveness_face/antispoofing_zalo/datasets/crops/scale_1.0/file_list.txt",
        params=params,
        model_path="G:/zalo_challenge/liveness_face/Silent-Face-Anti-Spoofing/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth",
        pre_trained=True,
    )

    deep_fake.fit(epochs=2)
