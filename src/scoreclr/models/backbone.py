import torch.nn as nn
import torchvision


def resnet(name: str, small_images: bool) -> nn.Module:
    """torchvision ResNet without its fc layer (`out_dim` features). For small images
    (CIFAR) the stem is a stride-1 3x3 conv without max-pooling."""
    net = getattr(torchvision.models, name)()
    net.out_dim = net.fc.in_features
    net.fc = nn.Identity()
    if small_images:
        net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
        net.maxpool = nn.Identity()
    return net
