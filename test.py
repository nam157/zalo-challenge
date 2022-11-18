from tqdm import tqdm

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix, F1Score

from load_data import CelebADataset
from models import MobileNet


torch.backends.cudnn.benchmark = True
torch.manual_seed(42)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT = 'ckpt/'


if __name__ == '__main__':
    dataset = CelebADataset(
        '/home/ai/datasets/CelebA_Spoof',
        'metas/intra_test/test_label.txt'
    )
    test_data_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)

    model = MobileNet().to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))

    criterion = CrossEntropyLoss()
    f1 = F1Score(num_classes=2).to(DEVICE)
    confmatrix = ConfusionMatrix(num_classes=2).to(DEVICE)

    with torch.no_grad():
        total_pred = torch.Tensor().to(DEVICE)
        total_label = torch.Tensor().to(DEVICE)
        test_loss = 0

        for img, label in tqdm(test_data_loader, desc="Testing"):
            img = img.to(DEVICE)
            label = label.to(DEVICE)
            label_argmax = torch.argmax(label, dim=1)

            pred = model(img)
            pred = torch.nn.functional.softmax(pred, dim=0)
            pred_argmax = torch.argmax(pred, dim=1)
            # pred_label = torch.zeros_like(pred).scatter_(1, pred_argmax.unsqueeze(1), 1.)

            loss = criterion(pred, label)

            total_pred = torch.cat((total_pred.to(torch.int8), pred_argmax.to(torch.int8)))
            total_label = torch.cat((total_label.to(torch.int8), label_argmax.to(torch.int8)))

            test_loss += loss.item()

        test_loss = test_loss / len(test_loss)
        test_acc = sum(total_label == total_label) / len(total_label)
        test_f1 = f1(total_pred, total_label)
        test_cfmatrix = confmatrix(total_pred, total_label)

    print('Test loss {:.6f} Acc {:.3f} F1 {:.3f}'.format(test_loss, test_acc, test_f1))
    print('Test confusion matrix:')
    print(test_cfmatrix)