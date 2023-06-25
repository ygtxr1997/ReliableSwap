import os.path
import random
import numpy as np
import cv2
import PIL.Image
import pickle


def main():
    """Demo of how to use the code"""

    # # path = sys.argv[1]

    # print("Loading occluders from Pascal VOC dataset...")
    # occluders = load_occluders(pascal_voc_root_path=path)
    with open(
            "/gavin/datasets/hanbang/hear/hands_voc.pkl",
            "rb"
    ) as handle:
        occluders = pickle.load(handle)
    print("Found {} suitable objects".format(len(occluders)))

    original_im = np.array(
        PIL.Image.open(
            "/gavin/datasets/original/hd_align_512/n000003/0139_02.jpg"
        ).resize((256, 256))
    )
    x, m = occlude_with_objects(original_im, occluders)
    PIL.Image.fromarray(x).save("test.png")
    PIL.Image.fromarray(m).save("m.png")


def load_occluders(pascal_voc_root_path="occluders.pkl"):
    import pickle

    with open(pascal_voc_root_path, "rb") as f:
        occluders = pickle.load(f)
    return occluders


def occlude_with_objects(im, occluders, count=5):
    """Returns an augmented version of `im`, containing some occluders from the Pascal VOC dataset."""
    width_height = np.asarray([im.shape[1], im.shape[0]])
    im_scale_factor = 1.0
    count = np.random.randint(1, count)
    mask_zeros = np.zeros_like(im)[..., 0]

    for _ in range(count):
        occluder = random.choice(occluders)
        random_scale_factor = np.random.uniform(0.5, 1.0)
        scale_factor = random_scale_factor * im_scale_factor
        occluder = resize_by_factor(occluder, scale_factor)
        center = np.random.uniform([0, 0], width_height)
        result, mask = paste_over(im_src=occluder, im_dst=im, center=center)
        mask_zeros = mask_zeros + mask
    mask_zeros = (np.clip(mask_zeros, 0, 1) * 255).astype(np.uint8)
    return result, mask_zeros


def paste_over(im_src, im_dst, center):
    """Pastes `im_src` onto `im_dst` at a specified position, with alpha blending, in place.
    Locations outside the bounds of `im_dst` are handled as expected (only a part or none of
    `im_src` becomes visible).
    Args:
        im_src: The RGBA image to be pasted onto `im_dst`. Its size can be arbitrary.
        im_dst: The target image.
        alpha: A float (0.0-1.0) array of the same size as `im_src` controlling the alpha blending
            at each pixel. Large values mean more visibility for `im_src`.
        center: coordinates in `im_dst` where the center of `im_src` should be placed.
    """
    mask_zeros = np.zeros_like(im_dst)[..., 0]

    hw = im_dst.shape[1]
    width_height_src = np.asarray([im_src.shape[1], im_src.shape[0]])
    width_height_dst = np.asarray([hw, hw])

    center = np.round(center).astype(np.int32)
    raw_start_dst = center - width_height_src // 2
    raw_end_dst = raw_start_dst + width_height_src

    start_dst = np.clip(raw_start_dst, 0, width_height_dst)
    end_dst = np.clip(raw_end_dst, 0, width_height_dst)
    region_dst = im_dst[start_dst[1]: end_dst[1], start_dst[0]: end_dst[0]]

    start_src = start_dst - raw_start_dst
    end_src = width_height_src + (end_dst - raw_end_dst)
    region_src = im_src[start_src[1]: end_src[1], start_src[0]: end_src[0]]
    color_src = region_src[..., 0:3]
    alpha = region_src[..., 3:].astype(np.float32) / 255

    mask_zeros[start_dst[1]: end_dst[1], start_dst[0]: end_dst[0]] = region_src[
        ..., -1
    ]

    im_dst[start_dst[1]: end_dst[1], start_dst[0]: end_dst[0]] = (
            alpha * color_src + (1 - alpha) * region_dst
    )
    return im_dst, mask_zeros


def resize_by_factor(im, factor):
    """Returns a copy of `im` resized by `factor`, using bilinear interp for up and area interp
    for downscaling.
    """
    new_size = tuple(
        np.round(np.array([im.shape[1], im.shape[0]]) * factor).astype(int)
    )
    interp = cv2.INTER_LINEAR if factor > 1.0 else cv2.INTER_AREA
    return cv2.resize(im, new_size, fx=factor, fy=factor, interpolation=interp)


def list_filepaths(dirpath):
    names = os.listdir(dirpath)
    paths = [os.path.join(dirpath, name) for name in names]
    return sorted(filter(os.path.isfile, paths))


if __name__ == "__main__":
    main()
