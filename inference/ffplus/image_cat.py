import os.path

from PIL import Image


def cat_cols(cols: list, start: int, end: int):
    n_row = len(cols[0])
    n_col = len(cols)
    w, h = cols[0][0].size

    n_row = end - start
    canvas = Image.new("RGB", (n_col * w, n_row * h, ))
    for i in range(n_row):
        for j in range(n_col):
            loc = (j * w, i * h)
            canvas.paste(cols[j][i + start], loc)

    return canvas


def cat_cols_and_save(cols: list, save_name: str, max_row_per_image: int = 200,):
    n_row = len(cols[0])
    n_col = len(cols)
    n_images = ((n_row - 1) // max_row_per_image) + 1

    for img_idx in range(n_images):
        start = img_idx * max_row_per_image
        end = min((img_idx + 1) * max_row_per_image, n_row)
        img_canvas = cat_cols(cols, start=start, end=end)
        img_canvas.save('%s_%03d.jpg' % (save_name.split('.')[0], img_idx))


def save_each_row(cols: list, save_folder: str):
    if os.path.exists(save_folder):
        os.system('rm -r %s' % save_folder)
    os.makedirs(save_folder, exist_ok=True)

    n_row = len(cols[0])
    n_col = len(cols)
    w, h = cols[0][0].size

    for i in range(n_row):
        one_row = Image.new("RGB", (n_col * w, h, ))
        for j in range(n_col):
            loc = (j * w, 0)
            one_row.paste(cols[j][i], loc)
        one_row.save(os.path.join(save_folder, '%04d.jpg' % i))


if __name__ == '__main__':
    import numpy as np

    t_list = []
    s_list = []
    r_list = []

    n = 1000
    for _ in range(n):
        t_list.append(Image.fromarray((np.random.rand(112, 112, 3) * 255).astype(np.uint8)))
        s_list.append(Image.fromarray((np.ones((112, 112, 3)) * 255).astype(np.uint8)))
        r_list.append(Image.fromarray((np.zeros((112, 112, 3)) * 255).astype(np.uint8)))

    cat_cols_and_save([t_list, s_list, r_list], save_name='tmp.jpg')
