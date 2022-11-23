import os
from tqdm import tqdm
import yaml

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset
from torchmetrics import ConfusionMatrix, F1Score

from sklearn.model_selection import KFold
import wandb

from load_data import CelebADataset, ZaloDataset, OnlyFakeDataset
from models import MobileNet, mobilevit_s
from utils import compute_eer


torch.backends.cudnn.benchmark = True
DEVICE =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WANDB = True
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
torch.manual_seed(config["random_state"])

if WANDB:
    wandb.init(
        project="zalo_challenge",
        config=config,
        name='ZaloTenCrop'
)


def hardcore_weight_sampler():
    base_weight = [1.78948, 2.2667]


if __name__ == '__main__':
    model = MobileNet()
    # model = mobilevit_s((224, 224), 2)
    model = model.to(DEVICE)
    # ckpt = torch.load("ckpt_vit_zalo/mobilenet_epoch_0_trainloss_0.686883_validloss_0.685381.pth", map_location=DEVICE)
    # model.load_state_dict(ckpt)

    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr=1e-3,
    #     momentum=0.9,
    #     nesterov=True,
    #     weight_decay=1e-6,
    # )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=1e-6
    )

    criterion = CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['T_max'], eta_min=config['eta_min'])

    os.makedirs(config["save_ckpt_dir"], exist_ok=True)
    
    f1 = F1Score(num_classes=2).to(DEVICE)
    confmatrix = ConfusionMatrix(num_classes=2).to(DEVICE)

    # dataset_celeb = CelebADataset(
    #     '/home/ai/datasets/CelebA_Spoof',
    #     'metas/intra_test/train_label.txt'
    # )
    dataset_zalo = ZaloDataset(
        image_path='/home/ai/datasets/challenge/liveness/generate/zalo_train',
        label_path='/home/ai/datasets/challenge/liveness/generate/zalo_train/face_crops.txt'
    )
    dataset_moire = OnlyFakeDataset(
        '/home/ai/datasets/challenge/liveness/fake_videos'
    )
    dataset = ConcatDataset([dataset_moire, dataset_zalo])
    # train_size = int(0.8 * len(dataset))
    # test_size = len(dataset) - train_size
    # data_train, data_val = torch.utils.data.random_split(dataset, [train_size, test_size])

    # train_data_loader = DataLoader(
    #     data_train,
    #     batch_size=config['batch_size'],
    #     shuffle=True,
    #     num_workers=config['num_workers'],
    #     pin_memory=True)
    # val_data_loader = DataLoader(data_val, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=True)

    kfold = KFold(n_splits=config['fold'], shuffle=True, random_state=config['random_state'])
    for epoch in range(config['epochs']):
        print('EPOCH {}/{}'.format(epoch+1, config['epochs']))

        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
            print('FOLD {}/{}'.format(fold+1, config['fold']))
            data_train = SubsetRandomSampler(train_ids)
            data_val = SubsetRandomSampler(test_ids)

            train_data_loader = DataLoader(dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], sampler=data_train, pin_memory=True)
            val_data_loader = DataLoader(dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], sampler=data_val, pin_memory=True)

            model.train()
            total_pred = torch.Tensor().to(DEVICE)
            total_label = torch.Tensor().to(DEVICE)
            train_loss = 0
            for img, label in tqdm(train_data_loader, desc="Training"):
                img = img.to(DEVICE)
                label = label.to(DEVICE)
                label_argmax = torch.argmax(label, dim=1)

                # bs, ncrops, c, h, w = img.size()
                # input_fit = img.view(-1, c, h, w)
                pred = model(img)
                # pred = model(input_fit)
                # output_avg = pred.view(bs, ncrops, -1).mean(1)

                # pred = torch.nn.functional.softmax(output_avg, dim=1)
                pred = torch.nn.functional.softmax(pred, dim=1)
                pred_argmax = torch.argmax(pred, dim=1)
                # pred_label = torch.zeros_like(pred).scatter_(1, pred_argmax.unsqueeze(1), 1.)

                loss = criterion(pred, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_pred = torch.cat((total_pred, pred_argmax))
                total_label = torch.cat((total_label, label_argmax))

                train_loss += loss.item()

            train_loss = train_loss / len(train_data_loader)
            train_acc = sum(total_label == total_label) / len(total_label)
            train_f1 = f1(total_pred.to(torch.int8), total_label.to(torch.int8))
            train_cfmatrix = confmatrix(total_pred.to(torch.int8), total_label.to(torch.int8))
            train_eer = compute_eer(total_label.detach().cpu().numpy(), total_pred.detach().cpu().numpy())

            print('Train loss {:.6f} Acc: {:.3f} F1 {:.3f} EER {:.3f}'.format(train_loss, train_acc, train_f1, train_eer))
            print('Train confusion matrix:')
            print(train_cfmatrix)
            
            if WANDB:
                wandb.log({
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'train_f1': train_f1,
                    'train_eer': train_eer,
                })

            with torch.no_grad():
                total_pred = torch.Tensor().to(DEVICE)
                total_label = torch.Tensor().to(DEVICE)
                val_loss = 0

                for img, label in tqdm(val_data_loader, desc="Validation"):
                    img = img.to(DEVICE)
                    label = label.to(DEVICE)
                    label_argmax = torch.argmax(label, dim=1)

                    # bs, ncrops, c, h, w = img.size()
                    # input_fit = img.view(-1, c, h, w)
                    pred = model(img)
                    # pred = model(input_fit)
                    # output_avg = pred.view(bs, ncrops, -1).mean(1)

                    # pred = torch.nn.functional.softmax(output_avg, dim=1)
                    pred = torch.nn.functional.softmax(pred, dim=1)
                    pred_argmax = torch.argmax(pred, dim=1)
                    # pred_label = torch.zeros_like(pred).scatter_(1, pred_argmax.unsqueeze(1), 1.)

                    loss = criterion(pred, label)

                    total_pred = torch.cat((total_pred.to(torch.int8), pred_argmax.to(torch.int8)))
                    total_label = torch.cat((total_label.to(torch.int8), label_argmax.to(torch.int8)))

                    val_loss += loss.item()

                val_loss = val_loss / len(val_data_loader)
                val_acc = sum(total_label == total_label) / len(total_label)
                val_f1 = f1(total_pred, total_label)
                val_cfmatrix = confmatrix(total_pred, total_label)
                val_eer = compute_eer(total_label.detach().cpu().numpy(), total_pred.detach().cpu().numpy())

            print('Valid loss {:.6f} Acc {:.3f} F1 {:.3f} EER {:.3f}'.format(val_loss, val_acc, val_f1, val_eer))
            print('Valid confusion matrix:')
            print(val_cfmatrix)
            if WANDB:
                wandb.log({
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'val_f1': val_f1,
                    'val_eer': val_eer,
                })
                wandb.log({
                    "valid_confusion_matrix": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=total_label.detach().cpu().numpy(), 
                        preds=total_pred.detach().cpu().numpy(),
                        class_names=['fake', 'real'])}
                )

            scheduler.step()

            torch.save(
                model.state_dict(),
                os.path.join(config['save_ckpt_dir'], 'mobilenet_epoch_{}_trainloss_{:.6f}_validloss_{:.6f}.pth'.format(
                    epoch, train_loss, val_loss
                )),
            )
