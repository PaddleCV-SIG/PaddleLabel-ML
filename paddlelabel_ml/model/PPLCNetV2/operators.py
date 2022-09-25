from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import random
from functools import partial

import cv2
import numpy as np
from PIL import Image


class Compose:
    """
    Compose preprocessing operators for obtaining prepocessed data. The shape of input image for all operations is [H, W, C], where H is the image height, W is the image width, and C is the number of image channels.
    Args:
        transforms(callmethod) : The method of preprocess images.
        to_rgb(bool): Whether to transform the input from BGR mode to RGB mode, default is False.
        channel_first(bool): whether to permute image from channel laste to channel first
    """

    def __init__(self, transforms, to_rgb: bool = True, channel_first: bool = True):
        if not isinstance(transforms, list):
            raise TypeError("The transforms must be a list!")
        if len(transforms) < 1:
            raise ValueError("The length of transforms " + "must be equal or larger than 1!")
        self.transforms = transforms
        self.to_rgb = to_rgb
        self.channel_first = channel_first

    def __call__(self, im):
        if isinstance(im, str):
            im = cv2.imread(im).astype("float32")

        if im is None:
            raise ValueError("Can't read The image file {}!".format(im))
        if self.to_rgb:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        for op in self.transforms:
            im = op(im)

        return im


class NormalizeImage(object):
    """normalize image such as substract mean, divide std"""

    def __init__(self, scale=None, mean=None, std=None, order="", output_fp16=False, channel_num=3):
        if isinstance(scale, str):
            scale = eval(scale)
        assert channel_num in [3, 4], "channel number of input image should be set to 3 or 4."
        self.channel_num = channel_num
        self.output_dtype = "float16" if output_fp16 else "float32"
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        self.order = order
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (3, 1, 1) if self.order == "chw" else (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype("float32")
        self.std = np.array(std).reshape(shape).astype("float32")

    def __call__(self, img):
        from PIL import Image

        if isinstance(img, Image.Image):
            img = np.array(img)

        assert isinstance(img, np.ndarray), "invalid input 'img' in NormalizeImage"

        img = (img.astype("float32") * self.scale - self.mean) / self.std

        if self.channel_num == 4:
            img_h = img.shape[1] if self.order == "chw" else img.shape[0]
            img_w = img.shape[2] if self.order == "chw" else img.shape[1]
            pad_zeros = np.zeros((1, img_h, img_w)) if self.order == "chw" else np.zeros((img_h, img_w, 1))
            img = (
                np.concatenate((img, pad_zeros), axis=0)
                if self.order == "chw"
                else np.concatenate((img, pad_zeros), axis=2)
            )
        return img.astype(self.output_dtype)


class CropImage(object):
    """crop image"""

    def __init__(self, size):
        if type(size) is int:
            self.size = (size, size)
        else:
            self.size = size  # (h, w)

    def __call__(self, img):
        w, h = self.size
        img_h, img_w = img.shape[:2]
        w_start = (img_w - w) // 2
        h_start = (img_h - h) // 2

        w_end = w_start + w
        h_end = h_start + h
        return img[h_start:h_end, w_start:w_end, :]


class OperatorParamError(ValueError):
    """OperatorParamError"""

    pass


class UnifiedResize(object):
    def __init__(self, interpolation=None, backend="cv2", return_numpy=True):
        _cv2_interp_from_str = {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "area": cv2.INTER_AREA,
            "bicubic": cv2.INTER_CUBIC,
            "lanczos": cv2.INTER_LANCZOS4,
            "random": (cv2.INTER_LINEAR, cv2.INTER_CUBIC),
        }
        _pil_interp_from_str = {
            "nearest": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "box": Image.BOX,
            "lanczos": Image.LANCZOS,
            "hamming": Image.HAMMING,
            "random": (Image.BILINEAR, Image.BICUBIC),
        }

        def _cv2_resize(src, size, resample):
            if isinstance(resample, tuple):
                resample = random.choice(resample)
            return cv2.resize(src, size, interpolation=resample)

        def _pil_resize(src, size, resample, return_numpy=True):
            if isinstance(resample, tuple):
                resample = random.choice(resample)
            if isinstance(src, np.ndarray):
                pil_img = Image.fromarray(src)
            else:
                pil_img = src
            pil_img = pil_img.resize(size, resample)
            if return_numpy:
                return np.asarray(pil_img)
            return pil_img

        if backend.lower() == "cv2":
            if isinstance(interpolation, str):
                interpolation = _cv2_interp_from_str[interpolation.lower()]
            # compatible with opencv < version 4.4.0
            elif interpolation is None:
                interpolation = cv2.INTER_LINEAR
            self.resize_func = partial(_cv2_resize, resample=interpolation)
        elif backend.lower() == "pil":
            if isinstance(interpolation, str):
                interpolation = _pil_interp_from_str[interpolation.lower()]
            self.resize_func = partial(_pil_resize, resample=interpolation, return_numpy=return_numpy)
        else:
            self.resize_func = cv2.resize

    def __call__(self, src, size):
        if isinstance(size, list):
            size = tuple(size)
        return self.resize_func(src, size)


class ResizeImage(object):
    """resize image"""

    def __init__(self, size=None, resize_short=None, interpolation=None, backend="cv2", return_numpy=True):
        if resize_short is not None and resize_short > 0:
            self.resize_short = resize_short
            self.w = None
            self.h = None
        elif size is not None:
            self.resize_short = None
            self.w = size if type(size) is int else size[0]
            self.h = size if type(size) is int else size[1]
        else:
            raise OperatorParamError(
                "invalid params for ReisizeImage for '\
                'both 'size' and 'resize_short' are None"
            )

        self._resize_func = UnifiedResize(interpolation=interpolation, backend=backend, return_numpy=return_numpy)

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img_h, img_w = img.shape[:2]
        else:
            img_w, img_h = img.size

        if self.resize_short is not None:
            percent = float(self.resize_short) / min(img_w, img_h)
            w = int(round(img_w * percent))
            h = int(round(img_h * percent))
        else:
            w = self.w
            h = self.h
        return self._resize_func(img, (w, h))


class ToCHWImage(object):
    """convert hwc image to chw image"""

    def __init__(self):
        pass

    def __call__(self, img):
        from PIL import Image

        if isinstance(img, Image.Image):
            img = np.array(img)

        return img.transpose((2, 0, 1))
