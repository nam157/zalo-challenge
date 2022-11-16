import os
from tqdm import tqdm

import torch
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from data import ContestDataset, CelebADataset
from model import MultiFTNet
from transform import transformer


def train(num_classes, img_channel, embedding_size, conv6_kernel, epochs, device):
    cls_criterion = CrossEntropyLoss()
    ft_criterion = MSELoss()

    param = {
        "num_classes": num_classes,
        "img_channel": img_channel,
        "embedding_size": embedding_size,
        "conv6_kernel": conv6_kernel,
    }

    model = MultiFTNet(**param)
    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000, eta_min=1e-7)

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
    # train_dataset = ContestDataset(
    #     label_list=label_train,
    #     transforms=transformer["train"],
    #     ft_height=ft_h,
    #     ft_width=ft_w,
    # )

    # test_dataset = ContestDataset(
    #     label_list=label_val,
    #     transforms=transformer['test'],
    #     ft_height=ft_h,
    #     ft_width=ft_w,
    # )
    kernel_size = get_kernel(128, 128)
    ft_h, ft_w = 2 * kernel_size[0], 2 * kernel_size[1]
    dataset = CelebADataset(
        '/home/ai/datasets/CelebA_Spoof',
        'metas/intra_test/test_label.txt',
        transformer['train'],
        ft_height=ft_h,
        ft_width=ft_w
    )
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    params = {"batch_size": 64, "shuffle": True, "num_workers": 8, "pin_memory": True}
    train_data_loader = DataLoader(train_dataset, **params)
    val_data_loader = DataLoader(test_dataset, **params)

    for e in range(epochs):
        print(f'Epoch {e}')

        total_train_loss = []
        total_val_loss = []

        for img,ft,label in tqdm(train_data_loader, desc="Training"):
            img = img.to(device)
            label = label.to(device)
            ft = ft.to(device)

            optimizer.zero_grad()
            embedding, feature_map = model(img)
            loss_cls = cls_criterion(embedding, label)
            loss_fea = ft_criterion(feature_map, ft)

            train_loss = 0.5 * loss_cls + 0.5 * loss_fea
            train_loss.backward()
            optimizer.step()

            total_train_loss.append(train_loss)

        log_train_loss = sum(total_train_loss)/len(total_train_loss)
        
        print('Train loss: {}'.format(log_train_loss.item()))
        
        with torch.no_grad():
            for img, ft, label in tqdm(val_data_loader, desc="Validation"):
                img = img.to(device)
                label = label.to(device)
                ft = ft.to(device)

                embedding, feature_map = model(img)
                loss_cls = cls_criterion(embedding, label)
                loss_fea = ft_criterion(feature_map, ft)

                val_loss = 0.5 * loss_cls + 0.5 * loss_fea

                total_val_loss.append(val_loss)

            log_val_loss = sum(total_val_loss)/len(total_val_loss)
            print('Val loss: {}'.format(log_val_loss.item()))

        torch.save(model, 'logs/model_epoch_{}.pth', e)
        torch.save(optimizer, 'logs/optimizer_epoch_{}.pth', e)

        scheduler.step()


if __name__ == "__main__":
    os.makedirs('logs/', exist_ok=True)

    def get_kernel(height, width):
        kernel_size = ((height + 15) // 16, (width + 15) // 16)
        return kernel_size

    train(
        num_classes=2,
        img_channel=3,
        embedding_size=128,
        conv6_kernel=get_kernel(128, 128),
        epochs=500,
        device=torch.device('cuda'),
    )
