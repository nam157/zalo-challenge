import os

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
from models import MobileNet

from sklearn.model_selection import KFold

from load_data import CelebADataset


torch.backends.cudnn.benchmark = True
torch.manual_seed(42)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 1
FOLD = 8


if __name__ == '__main__':
    dataset = CelebADataset(
        '/home/ai/datasets/CelebA_Spoof',
        'metas/intra_test/train_label.txt'
    )
    # train_size = int(0.8 * len(dataset))
    # test_size = len(dataset) - train_size
    # data_train, data_val = torch.utils.data.random_split(dataset, [train_size, test_size])

    # train_data_loader = DataLoader(data_train, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
    # val_data_loader = DataLoader(data_val, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)

    kfold = KFold(n_splits=FOLD, shuffle=True, random_state=42)
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        data_train = SubsetRandomSampler(train_ids)
        data_val = SubsetRandomSampler(test_ids)

        train_data_loader = DataLoader(dataset, batch_size=128, num_workers=8, sampler=data_train)
        val_data_loader = DataLoader(dataset, batch_size=128, num_workers=8, sampler=data_val)


    model = MobileNet().to(DEVICE)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-3,
        momentum=0.9,
        nesterov=True,
        weight_decay=1e-6,
    )
    criterion = CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30000, eta_min=1e-8)

    save_ckpt_dir = "ckpt/"
    os.makedirs(save_ckpt_dir, exist_ok=True)
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        print(f'EPOCH {epoch+1}/{EPOCHS}')
        for img, label in tqdm(train_data_loader, desc="Training"):
            img = img.to(DEVICE)
            label = label.to(DEVICE)

            pred = model(img)
            pred = torch.nn.functional.softmax(pred, dim=0)
            loss = criterion(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss = train_loss / len(train_data_loader)

        with torch.no_grad():
            val_loss = 0
            for img, label in tqdm(val_data_loader, desc="Validation"):
                img = img.to(DEVICE)
                label = label.to(DEVICE)

                pred = model(img)
                pred = torch.nn.functional.softmax(pred, dim=0)
                loss = criterion(pred, label)

                val_loss += loss.item()

            val_loss = val_loss / len(val_data_loader)

        print('Train loss {:.6f}\nValid loss {:.6f}'.format(train_loss, val_loss))

        scheduler.step()

        torch.save(
            model,
            os.path.join(save_ckpt_dir, 'mobilenet_epoch_{}_trainloss_{:.6f}_validloss_{:.6f}.pth'.format(
                epoch, train_loss, val_loss
            )),
        )
