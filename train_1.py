from model import MultiFTNet
from torch.nn import CrossEntropyLoss, MSELoss
import torch
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
from transform import transformer
from torch.utils.data import DataLoader
from data import Dataset
from tqdm import tqdm

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

    labels = open(
        "/home/ai/challenge/darf-nam/zalo-challenge/datasets/img_crops/scale_1/file_label.txt",
        "r",
    )
    data_labels = labels.readlines()
    label_train, label_val = train_test_split(
        data_labels, test_size=0.2, random_state=1
    )
    kernel_size = get_kernel(128, 128)
    ft_h, ft_w = 2 * kernel_size[0], 2 * kernel_size[1]
    data = Dataset(
        label_list=label_train,
        transforms=transformer["train"],
        ft_height=ft_h,
        ft_width=ft_w,
    )

    params = {"batch_size": 32, "shuffle": True, "num_workers": 8, "pin_memory": True}
    train_data_loader = DataLoader(data, **params)

    for e in tqdm(range(epochs)):
        print(f"Epoch: {e}")
        for img,ft,label in train_data_loader:
            img = img.to(device)
            label = label.to(device)
            ft = ft.to(device)

            optimizer.zero_grad()
            embedding, feature_map = model(img)
            loss_cls = cls_criterion(embedding, label)
            loss_fea = ft_criterion(feature_map, ft)

            loss = 0.5 * loss_cls + 0.5 * loss_fea
            loss.backward()
            optimizer.step()

            print(loss)


if __name__ == "__main__":

    def get_kernel(height, width):
        kernel_size = ((height + 15) // 16, (width + 15) // 16)
        return kernel_size

    train(
        num_classes=2,
        img_channel=3,
        embedding_size=128,
        conv6_kernel=get_kernel(128, 128),
        epochs=100,
        device=torch.device("cuda"),
    )
