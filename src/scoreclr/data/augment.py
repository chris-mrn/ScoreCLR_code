import random

from PIL import ImageFilter
from torchvision import transforms as T

IMAGENET_MEAN_STD = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
MEAN_STD = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}


class GaussianBlur:
    """PIL Gaussian blur with radius ~ U(0.1, 2.0)."""

    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 2.0)))


def normalize(dataset: str) -> T.Normalize:
    mean, std = MEAN_STD.get(dataset, IMAGENET_MEAN_STD)
    return T.Normalize(mean, std)


def pretrain_transform(dataset: str, crop_size: int, aug) -> T.Compose:
    """One augmented view: crop, color jitter, grayscale, blur, flip."""
    cj = aug.color_jitter
    return T.Compose(
        [
            T.RandomResizedCrop(crop_size, scale=tuple(aug.crop_scale), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomApply([T.ColorJitter(cj.brightness, cj.contrast, cj.saturation, cj.hue)], p=cj.prob),
            T.RandomGrayscale(p=aug.grayscale_prob),
            T.RandomApply([GaussianBlur()], p=aug.blur_prob),
            T.RandomHorizontalFlip(p=aug.flip_prob),
            T.ToTensor(),
            normalize(dataset),
        ]
    )


class TwoViews:
    """Returns two independent augmentations of the same image."""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        return self.transform(img), self.transform(img)


def eval_transforms(dataset: str, crop_size: int):
    """(train, test) transforms for the linear probe; test is also used for kNN / online accuracy."""
    train = [T.RandomResizedCrop(crop_size, scale=(0.08, 1.0)), T.RandomHorizontalFlip()]
    test = [] if crop_size <= 64 else [T.Resize(int(crop_size * 256 / 224)), T.CenterCrop(crop_size)]
    tail = [T.ToTensor(), normalize(dataset)]
    return T.Compose(train + tail), T.Compose(test + tail)
