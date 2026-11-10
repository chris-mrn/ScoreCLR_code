from PIL import Image

from scoreclr.data import TwoViews, eval_transforms, pretrain_transform
from scoreclr.utils import Config

AUG = Config(dict(
    crop_scale=[0.08, 1.0], grayscale_prob=0.2, blur_prob=0.5, flip_prob=0.5,
    color_jitter=dict(prob=0.8, brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
))


def test_two_views_differ():
    img = Image.new("RGB", (40, 40), color=(120, 30, 200))
    img.putpixel((3, 3), (0, 0, 0))
    v0, v1 = TwoViews(pretrain_transform("cifar10", 32, AUG))(img)
    assert v0.shape == v1.shape == (3, 32, 32)


def test_eval_transforms_sizes():
    img = Image.new("RGB", (300, 260))
    train_tf, test_tf = eval_transforms("imagenet100", 224)
    assert train_tf(img).shape == test_tf(img).shape == (3, 224, 224)
