import torch
import math
import random
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from up.data.datasets.transforms import Augmentation
from up.utils.general.registry_factory import AUGMENTATION_REGISTRY, BATCHING_REGISTRY
from up.tasks.cls.data.rand_augmentation import *
from up.tasks.cls.data.auto_augmentation import *
from up.tasks.cls.data.data_utils import *
import copy

try:
    from torchvision.transforms.functional import InterpolationMode
    _torch_interpolation_to_str = {InterpolationMode.NEAREST: 'nearest',
                                   InterpolationMode.BILINEAR: 'bilinear',
                                   InterpolationMode.BICUBIC: 'bicubic',
                                   InterpolationMode.BOX: 'box',
                                   InterpolationMode.HAMMING: 'hamming',
                                   InterpolationMode.LANCZOS: 'lanczos'}
except: # noqa
    _torch_interpolation_to_str = {0: 'nearest',
                                   1: 'lanczos',
                                   2: 'bilinear',
                                   3: 'bicubic',
                                   4: 'box',
                                   5: 'hamming'}
_str_to_torch_interpolation = {b: a for a, b in _torch_interpolation_to_str.items()}


__all__ = [
    'TorchAugmentation',
    'RandomResizedCrop',
    'RandomHorizontalFlip',
    'PILColorJitter',
    'TorchResize',
    'TorchCenterCrop',
    'RandomErasing',
    'RandAugment',
    'RandAugIncre',
    'AutoAugment']

############################ add ############
from PIL import Image, ImageFilter
import skimage

@AUGMENTATION_REGISTRY.register('torch_random_noise')
class RandomNoise:
    """按概率为图像添加噪声"""
    def __init__(self, modes=['gaussian'], p=0.15):
        """
        Params:
        modes: list or tuple of strings
            添加噪声的类型，如 'gaussian', 'localvar', 'poisson', 'salt', 'pepper', 's&p', 'speckle'
        p: float
            执行该操作的概率
        """
        self.modes = modes
        self.p = p

    def __call__(self, data):
        output = copy.copy(data)
        image = data.image
        if random.uniform(0, 1) < self.p:  # 按概率执行该操作
            img_arr = np.array(image)
            for mode in self.modes:
                img_arr_noise = skimage.util.random_noise(img_arr, mode)
            img_pil = Image.fromarray((img_arr*0.5+(img_arr_noise*0.5)* 255).astype(np.uint8))
            output.image = img_pil
        else:
            output.image = image
        return output

@AUGMENTATION_REGISTRY.register('torch_gaussion_blur')
class GaussionBlur:
    def __init__(self, p=0.15):
        self.p = p

    def __call__(self, data):
        output = copy.copy(data)
        image = data.image
        if random.uniform(0, 1) < self.p:  # 按概率执行该操作
            output.image = image.filter(ImageFilter.GaussianBlur(radius=1))
        else:
            output.image = image
        return output

@AUGMENTATION_REGISTRY.register('torch_random_pad')
class RandomPad:
    """按概率为图像添加噪声"""
    def __init__(self, maxratio=0.2):
        self.maxratio = maxratio

    def __call__(self, data):
        output = copy.copy(data)
        image = data.image

        h,w,_ = np.array(image).shape
        pad_h = random.randint(-int(h * self.maxratio),int(h * self.maxratio))
        pad_w = random.randint(-int(w * self.maxratio),int(w * self.maxratio))
        pad_h = pad_h if pad_h>0 else 0
        pad_w = pad_w if pad_w>0 else 0
        pad = transforms.Pad(padding=(pad_w,pad_h), padding_mode='edge')
        output.image = pad(image)
        return output

@AUGMENTATION_REGISTRY.register('torch_random_affine')
class RandomAffine:
    def __init__(self, maxratio=0):
        self.maxratio = maxratio

    def __call__(self, data):
        output = copy.copy(data)
        image = data.image
        trans = transforms.RandomAffine(8, translate=[0.2, 0.2], scale=[0.8, 1.2], shear=(-8, 8, -8, 8), interpolation=2)
        # trans = transforms.RandomAffine(5, translate=[0.15, 0.15], scale=[0.8, 1.2], shear=(-5, 5, -5, 5), interpolation=2)

        output.image = trans(image)
        return output

@AUGMENTATION_REGISTRY.register('torch_random_resized_crop_wh_filter')
class RandomResizedCropWhFilter:
    def __init__(self, size, scale=[0.08,1],ratio=[0.8,1.1]):
        self.size = size
        self.scale = scale
        self.ratio = ratio

    def __call__(self, data):
        output = copy.copy(data)
        image = data.image
        h, w, _ = np.array(image).shape
        if torch.rand(1) < 0.1:
            trans1 = transforms.Resize([28, 28])
            output.image = trans1(image)
            return output
        else:
        # if h/w<0.3 or h/w>2.5:
        #     trans1 = transforms.Resize([128, 128])
        #     trans2 = transforms.RandomResizedCrop(112, scale=(0.4, 1.0))
        #     image = trans1(image)
        #     output.image = trans2(image)
        # else:
            trans = transforms.RandomResizedCrop(self.size, scale=self.scale,ratio = self.ratio)
            output.image = trans(image)
        return output

##############################################

class TorchAugmentation(Augmentation):
    def __init__(self):
        self.op = lambda x: x

    def augment(self, data):
        output = copy.copy(data)
        output.image = self.op(data.image)
        return output


@AUGMENTATION_REGISTRY.register('torch_random_resized_crop')
class RandomResizedCrop(TorchAugmentation):
    def __init__(self, size, **kwargs):
        if 'interpolation' in kwargs:
            kwargs['interpolation'] = _str_to_torch_interpolation[kwargs['interpolation']]
        self.op = transforms.RandomResizedCrop(size, **kwargs)


# @AUGMENTATION_REGISTRY.register('torch_random_horizontal_flip')
# class RandomHorizontalFlip(TorchAugmentation):
#     def __init__(self, p=0.5):
#         self.p = p
#
#     def augment(self, data):
#         output = copy.copy(data)
#         if torch.rand(1) < self.p:
#             output.image = TF.hflip(data.image)
#         return output

@AUGMENTATION_REGISTRY.register('torch_random_horizontal_flip')
class RandomHorizontalFlip(TorchAugmentation):
    def __init__(self, p=0.5):
        self.p = p
        self.hflip_cls = [15, 43, 44, 45, 56, 58, 60, 70, 71, 72, 75, 77, 80, 82, 85, 87, 88]
        self.vflip_cls = [15, 43, 44, 45, 70, 73, 74, 75, 78, 79, 80, 83, 84, 85, 87, 88]
        self.rotate_cls = [15, 43, 45, 70, 75, 80, 85, 87, 88]

    def augment(self, data):
        output = copy.copy(data)
        if torch.rand(1) < self.p and (data.gt in self.hflip_cls):
            output.image = TF.hflip(data.image)
        if torch.rand(1) < self.p and (data.gt in self.vflip_cls):
            output.image = TF.vflip(output.image)
        if torch.rand(1) < self.p and (data.gt in self.rotate_cls):
            output.image = TF.rotate(output.image, 90, expand=True)
        return output

@AUGMENTATION_REGISTRY.register('torch_color_jitter')
class PILColorJitter(TorchAugmentation):
    def __init__(self, brightness, contrast, saturation, hue=0.):
        self.op = transforms.ColorJitter(brightness, contrast, saturation, hue)


@AUGMENTATION_REGISTRY.register('torch_resize')
class TorchResize(TorchAugmentation):
    def __init__(self, size, **kwargs):
        if 'interpolation' in kwargs:
            kwargs['interpolation'] = _str_to_torch_interpolation[kwargs['interpolation']]
        self.op = transforms.Resize(size, **kwargs)


@AUGMENTATION_REGISTRY.register('torch_center_crop')
class TorchCenterCrop(TorchAugmentation):
    def __init__(self, size, **kwargs):
        self.op = transforms.CenterCrop(size, **kwargs)


@AUGMENTATION_REGISTRY.register('torch_auto_augmentation')
class AutoAugment(TorchAugmentation):
    def __init__(self, size, **kwargs):
        self.op = ImageNetPolicy()


@AUGMENTATION_REGISTRY.register('torch_random_augmentation')
class RandAugment(TorchAugmentation):
    def __init__(self, n, m, magnitude_std=0.0):
        self.n = n
        self.m = m
        self.augment_list = augment_list()
        self.mstd = magnitude_std

    def augment(self, data):
        output = copy.copy(data)
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            if random.random() > 0.5:
                continue
            magnitude = random.gauss(self.m, self.mstd)
            magnitude = max(0, min(magnitude, 10))
            val = (float(magnitude) / 10) * float(maxval - minval) + minval
            output.image = op(output.image, val)

        return output


@AUGMENTATION_REGISTRY.register('torch_random_augmentationIncre')
class RandAugIncre(TorchAugmentation):
    def __init__(self, n, m, magnitude_std=0.0):
        self.n = n
        self.m = m
        self.augment_list = augment_increasing()
        self.mstd = magnitude_std

    def augment(self, data):
        output = copy.copy(data)
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            if random.random() > 0.5:
                continue
            magnitude = random.gauss(self.m, self.mstd)
            magnitude = max(0, min(magnitude, 10))
            val = (float(magnitude) / 10) * float(maxval - minval) + minval
            output.image = op(output.image, val)

        return output


@AUGMENTATION_REGISTRY.register('torch_randerase')
class RandomErasing(TorchAugmentation):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
        This variant of RandomErasing is intended to be applied to either a batch
        or single image tensor after it has been normalized by dataset mean and std.
    Args:
         probability: Probability that the Random Erasing operation will be performed.
         min_area: Minimum percentage of erased area wrt input image area.
         max_area: Maximum percentage of erased area wrt input image area.
         min_aspect: Minimum aspect ratio of erased area.
         mode: pixel color mode, one of 'const', 'rand', or 'pixel'
            'const' - erase block is constant color of 0 for all channels
            'rand'  - erase block is same per-channel random (normal) color
            'pixel' - erase block is per-pixel random (normal) color
        max_count: maximum number of erasing blocks per image, area per box is scaled by count.
            per-image count is randomly chosen between 1 and this value.
    """

    def __init__(
            self,
            probability=0.5, min_area=0.02, max_area=1 / 3, min_aspect=0.3, max_aspect=None,
            mode='pixel', min_count=1, max_count=None, num_splits=0, device='cpu'):
        self.probability = probability
        self.min_area = min_area
        self.max_area = max_area
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        self.min_count = min_count
        self.max_count = max_count or min_count
        self.num_splits = num_splits
        self.mode = mode.lower()
        self.rand_color = False
        self.per_pixel = False
        if self.mode == 'rand':
            self.rand_color = True  # per block random normal
        elif self.mode == 'pixel':
            self.per_pixel = True  # per pixel random normal
        else:
            assert not self.mode or self.mode == 'const'
        self.device = device

    def _erase(self, img, chan, img_h, img_w, dtype):
        if random.random() > self.probability:
            return
        area = img_h * img_w
        count = self.min_count if self.min_count == self.max_count else \
            random.randint(self.min_count, self.max_count)
        for _ in range(count):
            for attempt in range(10):
                target_area = random.uniform(self.min_area, self.max_area) * area / count
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < img_w and h < img_h:
                    top = random.randint(0, img_h - h)
                    left = random.randint(0, img_w - w)
                    img[:, top:top + h, left:left + w] = self._get_pixels(
                        self.per_pixel, self.rand_color, (chan, h, w),
                        dtype=dtype, device=self.device)
                    break

    def augment(self, data):
        image = copy.copy(data).image
        self._erase(image, *image.size(), image.dtype)
        data.image = image
        return data

    def _get_pixels(self, per_pixel, rand_color, patch_size, dtype=torch.float32, device='cuda'):
        if per_pixel:
            return torch.empty(patch_size, dtype=dtype, device=device).normal_()
        elif rand_color:
            return torch.empty((patch_size[0], 1, 1), dtype=dtype, device=device).normal_()
        else:
            return torch.zeros((patch_size[0], 1, 1), dtype=dtype, device=device)

    def __repr__(self):
        fs = self.__class__.__name__ + f'(p={self.probability}, mode={self.mode}'
        fs += f', count=({self.min_count}, {self.max_count}))'
        return fs


@BATCHING_REGISTRY.register('batch_mixup')
class BatchMixup(object):
    def __init__(self, alpha=1.0, num_classes=1000):
        self.alpha = alpha
        self.num_classes = num_classes

    def __call__(self, data):
        """
        Args:
            images: list of tensor
        """
        return mixup(data, self.alpha, self.num_classes)


@BATCHING_REGISTRY.register('batch_cutmix')
class BatchCutMix(object):
    def __init__(self, alpha=1.0, num_classes=1000):
        self.alpha = alpha
        self.num_classes = num_classes

    def __call__(self, data):
        """
        Args:
            images: list of tensor
        """
        return cutmix(data, self.alpha, self.num_classes)


@BATCHING_REGISTRY.register('batch_cutmixup')
class BatchCutMixup(object):
    def __init__(self, mixup_alpha=1.0, cutmix_alpha=1.0, switch_prob=0.5, num_classes=1000):
        self.switch_prob = switch_prob
        self.num_classes = num_classes
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha

    def __call__(self, data):
        use_cutmix = np.random.rand() < self.switch_prob
        data.gt = data.gt.long()
        if use_cutmix:
            return cutmix(data, self.cutmix_alpha, self.num_classes)

        return mixup(data, self.mixup_alpha, self.num_classes)
