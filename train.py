import os
import warnings

import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from torch.utils import data

import utils
from load_data import Data
from models.net import Net

warnings.filterwarnings("ignore")


def train(net, dataloader, optimizer, epoch):
    net.train()
    scores = []
    all_y = []

    for e in range(epoch):
        for batch_idx, (X, y) in enumerate(dataloader):

            X, y = X.to(device), y.to(device).view(-1,)
            optimizer.zero_grad()
            output = net(X)
            scores.extend(F.softmax(output).detach().cpu().numpy()[:, 1:])
            all_y.extend(y.cpu().numpy())
            loss = F.cross_entropy(output, y)
            loss.backward()
            optimizer.step()

        if e % 5 == 0:
            print("the loss is", loss.item())
            torch.save(net, f"ckpt/model_scale_{2.7}.pth")
            val(net, test_data_loader, e)


def val(net, dataloader, e):
    net.eval()
    scores = []
    all_y = []
    for batch_idx, (X, y) in enumerate(dataloader):

        X, y = X.to(device), y.to(device).view(-1,)
        optimizer.zero_grad()
        output = net(X)
        scores.extend(F.softmax(output).detach().cpu().numpy()[:, 1:])
        all_y.extend(y.cpu().numpy())

    print("epoch %d " % (e))
    utils.EER(all_y, scores)
    utils.HTER(all_y, scores, 0.5)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # Detect devices
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU
    learning_rate = 0.01

    params = (
        {"batch_size": 8, "shuffle": True, "num_workers": 0, "pin_memory": True}
        if use_cuda
        else {}
    )
    if use_cuda:
        net = Net(2).to(device)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(net)

    dataset = Data(
        "G:/zalo_challenge/liveness_face/antispoofing_zalo/datasets/crops/scale_2.7/",
        True,
    )
    train_data_set, valid_data_set = train_test_split(
        dataset, test_size=0.2, random_state=1
    )
    train_data_loader = data.DataLoader(train_data_set, **params)
    test_data_loader = data.DataLoader(valid_data_set, **params)
    optimizer = torch.optim.SGD(list(net.parameters()), lr=learning_rate)

    train(net, dataloader=train_data_loader, optimizer=optimizer, epoch=100)
