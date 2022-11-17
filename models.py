import torch
from torchvision.models import mobilenet_v2


class MobileNet(torch.nn.Module):
    def __init__(self, num_classes=2, pretrained=False) -> None:
        super().__init__()
        self.model = mobilenet_v2(num_classes=num_classes, pretrained=pretrained)

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    model = MobileNet()