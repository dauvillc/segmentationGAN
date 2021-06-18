"""
Defines some image manipulation and saving functions.
"""

import numpy as np
from skimage.io import imsave


def convert_to_8bits_rgb(img):
    """
    Converts an image of shape (channels, h, w)
    to an image of shape (3, h, w) whose dtype
    is uint8.
    """
    img = (img * 255 / np.max(img)).astype(np.uint8)
    if len(img.shape) == 3 and img.shape[0] == 2:
        img = img[0]
    if len(img.shape) == 2:
        img = np.repeat(np.expand_dims(img, 1), 3, axis=1)

    return np.moveaxis(img, 1, -1)
