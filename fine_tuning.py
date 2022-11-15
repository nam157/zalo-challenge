import os
from typing import Dict

import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from load_data import Dataset, transformer
from models.MiniFASNet import (MiniFASNetV1, MiniFASNetV1SE, MiniFASNetV2,
                               MiniFASNetV2SE)
from utils import get_kernel, parse_model_name

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


class anti_spoofing:
    def __init__(
        self,
        label_path: str,
        params: Dict,
        model_path: str = None,
        pre_trained: bool = False,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if pre_trained:
            model_name = os.path.basename(model_path)
            h_input, w_input, model_type, _ = parse_model_name(model_name)
            self.kernel_size = get_kernel(
                h_input,
                w_input,
            )
            self.model = MODEL_MAPPING[model_type](conv6_kernel=self.kernel_size).to(
                self.device
            )
            self.model = torch.nn.DataParallel(self.model)
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            self.kernel_size = get_kernel(80, 80)
            self.model = MODEL_MAPPING[MODEL_TYPE](conv6_kernel=self.kernel_size).to(
                self.device
            )

        self.train_data_loader, self.val_data_loader = self._init_data(
            label_path, params
        )
        (
            self.optimizer,
            self.scheduler,
            self.cls_criterion,
        ) = self._init_params()

        self.writer = SummaryWriter()

    def _init_data(self, label_path: str, params: Dict):
        fh = open(label_path, "r")
        labels = fh.readlines()
        label_train, label_valid = train_test_split(
            labels, test_size=0.1, random_state=123
        )
        data_train = Dataset(
            label_list=label_train,
            transforms=transformer["train"],
        )
        data_val = Dataset(
            label_list=label_valid,
            transforms=transformer["test"],
        )

        train_data_loader = DataLoader(data_train, **params)
        val_data_loader = DataLoader(data_val, **params)
        return train_data_loader, val_data_loader

    def _init_params(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=0.0001,
            momentum=0.9,
            nesterov=True,
            weight_decay=0.000001,
        )
        # optimizer = torch.optim.Adam(
        #     self.model.parameters(), lr=0.0001, amsgrad=True, weight_decay=0.00001
        # )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1
        )
        cls_criterion = CrossEntropyLoss()

        return optimizer, scheduler, cls_criterion

    def _train_one_epoch(self, device):
        self.model.train()
        loss_sum = 0
        for img, label in tqdm(self.train_data_loader, desc="Training"):
            img = img.to(device)
            label = label.to(device)
            self.optimizer.zero_grad()
            out = self.model(img)
            train_loss = self.cls_criterion(out, label)

            train_loss.backward()
            self.optimizer.step()

            loss_sum += train_loss
        return loss_sum / len(self.train_data_loader)

    def _valid_one_epoch(self, device):
        self.model.eval()
        loss_sum = 0
        for img, label in tqdm(self.val_data_loader, desc="Validation"):
            with torch.no_grad():
                img = img.to(device)
                label = label.to(device)

                out = self.model(img)
                valid_loss = self.cls_criterion(out, label)

                loss_sum += valid_loss
        return loss_sum / len(self.val_data_loader)

    def fit(self, epochs):

        for epoch in range(epochs):
            loss_train = self._train_one_epoch(device=self.device)
            self.scheduler.step(loss_train)
            loss_val = self._valid_one_epoch(device=self.device)
            lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("Train/Loss", loss_train, epoch)
            self.writer.add_scalar("Valid/Loss", loss_val, epoch)
            self.writer.add_scalar("Train/Learning_rate", lr, epoch)
            print(
                f"Epochs: {epoch}/{epochs}--- Loss-Train: {loss_train} ----- Loss-Valid: {loss_val}"
            )
            if epoch % 5 == 0:
                save_dir = "ckpt/anti_spoofing/"
                os.makedirs(save_dir, exist_ok=True)
                torch.save(
                    self.model.state_dict(),
                    save_dir + f"{SCALE}_{HEIGHT}x{WIDTH}_{MODEL_TYPE}.pth",
                )


if __name__ == "__main__":
    params = {"batch_size": 128, "shuffle": True, "num_workers": 8, "pin_memory": True}
    deep_fake = anti_spoofing(
        label_path="/home/ai/challenge/datasets/crops_80x80/scale_2.7/file_list.txt",
        params=params,
        model_path="./pre-trained/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth",
        pre_trained=False,
    )
    deep_fake.fit(epochs=100)
