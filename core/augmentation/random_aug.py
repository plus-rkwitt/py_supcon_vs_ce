import torchvision.transforms as T
import numpy as np


from PIL import ImageOps, ImageEnhance

"""
Some image operations implemented in torchvision.
"""

translate_x = T.RandomAffine(degrees=0, translate=(0.25, 0))

translate_y = T.RandomAffine(degrees=0, translate=(0, 0.25))

scale = T.RandomAffine(degrees=0, scale=(0.5, 1.5))

shear_x = T.RandomAffine(degrees=0, shear=(-20, 20, 0, 0))

shear_y = T.RandomAffine(degrees=0, shear=(0, 0, -20, 20))

rotate = T.RandomAffine(degrees=(-30, 30))

flip = T.RandomHorizontalFlip()

crop = T.RandomCrop(32, padding=4)


"""
Other image operations implemented by PIL
"""

def auto_contrast(img):
    return ImageOps.autocontrast(img)


def invert(img):
    return ImageOps.invert(img)


def equalize(img):
    return ImageOps.equalize(img)


def solarize(img):
    return ImageOps.solarize(img, np.random.uniform(0, 256))


def posterize(img):
    return ImageOps.posterize(img, np.random.randint(0, 7))


def contrast(img):
    return ImageEnhance.Contrast(img).enhance(np.random.rand()*2.0)


def color(img):
    return ImageEnhance.Color(img).enhance(np.random.rand()*2.0)


def brightness(img):
    return ImageEnhance.Brightness(img).enhance(np.random.rand()*2.0)


def sharpness(img):
    return ImageEnhance.Sharpness(img).enhance(np.random.rand()*2.0)


"""
Augmenter
"""
class RandomAugment(object):
    def __init__(self):
        """
        List of potential operations
        """
        self.transforms = [
            translate_x,
            translate_y,
            scale,
            shear_x,
            shear_y,
            rotate,
            auto_contrast,
            invert,
            equalize,
            solarize,
            posterize,
            contrast,
            color,
            brightness,
            sharpness,
            flip,
            crop
        ]

    def __call__(self, img):
        """
        A policy consists of n operations (here: 1, or 2)

        Depending on the length, n, of a policy we randomly apply n operations.
        """
        n = np.random.randint(1, 3)
        for _ in range(n):
            t = self.transforms[np.random.randint(len(self.transforms))]
            img = t(img)

        return img
