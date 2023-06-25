"""image related ops."""
import math

import numpy as np
import cv2
import tensorflow._api.v2.compat.v1 as tf


def np_scale_to_shape(image, shape, align=True):
    """Scale the image.

    The minimum side of height or width will be scaled to or
    larger than shape.

    Args:
        image: numpy image, 2d or 3d
        shape: (height, width)

    Returns:
        numpy image
    """
    height, width = shape
    imgh, imgw = image.shape[0:2]
    if imgh < height or imgw < width or align:
        scale = np.maximum(height/imgh, width/imgw)
        image = cv2.resize(
            image,
            (math.ceil(imgw*scale), math.ceil(imgh*scale)))
    return image


def np_random_crop(image, shape, random_h=None, random_w=None, align=True):
    """Random crop.

    Shape from image.

    Args:
        image: Numpy image, 2d or 3d.
        shape: (height, width).
        random_h: A random int.
        random_w: A random int.

    Returns:
        numpy image
        int: random_h
        int: random_w

    """
    height, width = shape
    image = np_scale_to_shape(image, shape, align=align)
    imgh, imgw = image.shape[0:2]
    if random_h is None:
        random_h = np.random.randint(imgh-height+1)
    if random_w is None:
        random_w = np.random.randint(imgw-width+1)
    return (image[random_h:random_h+height, random_w:random_w+width, :],
            random_h, random_w)
