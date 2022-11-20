import os
import warnings

import torch
from torch import optim
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from method_evaluate import get_equal_error_rate, get_tp_fp_rates
from src.data_io.dataset_loader import get_train_loader, get_val_loader
from src.model_lib.MiniFASNet import (
    MiniFASNetV1,
    MiniFASNetV1SE,
    MiniFASNetV2,
    MiniFASNetV2SE,
)
from src.model_lib.MultiFTNet import MultiFTNet
from src.utility import get_time

warnings.filterwarnings("ignore")

MODEL_MAPPING = {
    "MiniFASNetV2": MiniFASNetV2,
    "MiniFASNetV1": MiniFASNetV1,
    "MiniFASNetV1SE": MiniFASNetV1SE,
    "MiniFASNetV2SE": MiniFASNetV2SE,
}


class TrainMain:
    def __init__(self, conf):
        self.conf = conf
        self.board_loss_every = conf.board_loss_every
        self.save_every = conf.save_every
        self.step = 0
        self.start_epoch = 0
        self.train_loader = get_train_loader(self.conf)
        self.valid_loader = get_val_loader(self.conf)

    def train_model(self):
        self._init_model_param()
        self._train_stage()

    def _init_model_param(self):
        self.cls_criterion = CrossEntropyLoss()
        self.ft_criterion = MSELoss()
        self.model = self._define_network()
        self.optimizer = optim.SGD(
            self.model.module.parameters(),
            lr=self.conf.lr,
            weight_decay=5e-4,
            momentum=self.conf.momentum,
        )

        if self.conf.schedule_type == "MultiStepLR":
            self.schedule_lr = optim.lr_scheduler.MultiStepLR(
                self.optimizer, self.conf.milestones, self.conf.gamma, -1
            )
        elif self.conf.schedule_type == "MultiStepLR":
            self.schedule_lr = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.01, verbose=True, patience=5
            )
        else:
            self.schedule_lr = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=10, eta_min=0.00000001, verbose=True
            )

        print("lr: ", self.conf.lr)
        print("epochs: ", self.conf.epochs)
        print("milestones: ", self.conf.milestones)
        print("model_type: ", self.conf.model_type)
        # print("pre-trained", os.path.basename(self.conf.pre_trained))

    def _train_stage(self):
        self.writer = SummaryWriter()
        for e in range(self.start_epoch, self.conf.epochs):
            print(f"--------Epoch----------: {e}")
            loss_train, loss_cls, loss_fea, acc, eer_train = self._train_batch_data()
            loss_val, eer_val, acc_val = self._valid_batch_data()

            # Log
            self.writer.add_scalar("Training/Loss", loss_train, e)
            self.writer.add_scalar("Training/Acc", acc, e)
            self.writer.add_scalar(
                "Training/Lr", self.optimizer.param_groups[0]["lr"], e
            )
            self.writer.add_scalar("Training/Loss_cls", loss_cls, e)
            self.writer.add_scalar("Training/Loss_ft", loss_fea, e)
            self.writer.add_scalar("Training/EER", eer_train, e)

            self.writer.add_scalar("Validation/Loss", loss_val, e)
            self.writer.add_scalar("Validation/EER", eer_val, e)
            self.writer.add_scalar("Validation/ACC", acc_val, e)

            torch.save(self.model.state_dict(), self.conf.model_path + f"_{e}.pth")

        if isinstance(self.conf.schedule_type, optim.lr_scheduler.ReduceLROnPlateau):
            self.schedule_lr.step(loss_train)
        else:
            self.schedule_lr.step()
        self.writer.close()

    def _train_batch_data(self):
        self.model.train()
        loss_sum = 0
        loss_cls_sum = 0
        loss_fea_sum = 0
        acc_sum = 0
        eer_sum = 0
        for imgs, ft_sample, target in tqdm(iter(self.train_loader)):
            self.optimizer.zero_grad()
            target = target.to(self.conf.device)
            embeddings, feature_map = self.model.forward(imgs.to(self.conf.device))

            loss_cls = self.cls_criterion(embeddings, target)
            loss_fea = self.ft_criterion(feature_map, ft_sample.to(self.conf.device))
            loss = 0.5 * loss_cls + 0.5 * loss_fea

            loss_sum += loss
            loss_cls_sum += loss_cls
            loss_fea_sum += loss_fea

            loss.backward()
            self.optimizer.step()
            acc = self._get_accuracy(embeddings, target)[0]
            acc_sum += acc
            
            fpr, tpr, threshold = get_tp_fp_rates(target.detach().cpu().numpy(), embeddings)
            
            eer = get_equal_error_rate(tpr=tpr, fpr=fpr)
            eer_sum += eer

        return (
            loss_sum / len(self.train_loader),
            loss_cls_sum / len(self.train_loader),
            loss_fea_sum / len(self.train_loader),
            acc_sum / len(self.train_loader),
            eer_sum / len(self.train_loader),
        )

    def _valid_batch_data(self):
        self.model.eval()
        loss = 0
        eer_sum = 0
        acc_sum = 0
        for imgs, ft_sample, target in tqdm(iter(self.valid_loader)):
            target = target.to(self.conf.device)
            embeddings = self.model.forward(imgs.to(self.conf.device))
            loss_cls = self.cls_criterion(embeddings, target)
            loss += loss_cls
            acc = self._get_accuracy(embeddings, target)[0]
            acc_sum += acc
            
            fpr, tpr, threshold = get_tp_fp_rates(target.detach().cpu().numpy(), embeddings)
            eer = get_equal_error_rate(tpr=tpr, fpr=fpr)
            eer_sum += eer
        return (
            loss / len(self.valid_loader),
            eer_sum / len(self.valid_loader),
            acc_sum / len(self.valid_loader),
        )

    def _define_network(self):
        param = {
            "model_type": MODEL_MAPPING[self.conf.model_type],
            "num_classes": self.conf.num_classes,
            "img_channel": self.conf.input_channel,
            "embedding_size": self.conf.embedding_size,
            "conv6_kernel": self.conf.kernel_size,
            "training": True,
            "pre_trained": self.conf.pre_trained,
        }

        model = MultiFTNet(**param).to(self.conf.device)
        model = torch.nn.DataParallel(model, self.conf.devices)
        model.to(self.conf.device)
        return model

    def _get_accuracy(self, output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        ret = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)
            ret.append(correct_k.mul_(1.0 / batch_size))
        return ret

    def _save_state(self, time_stamp, extra=None):
        save_path = self.conf.model_path
        torch.save(
            self.model.state_dict(),
            save_path
            + "/"
            + ("{}_{}_model_iter-{}.pth".format(time_stamp, extra, self.step)),
        )
