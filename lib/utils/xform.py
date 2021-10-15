import cv2
import numpy as np

# while this code logically belongs in transforms.py, I put it in
# its own module to minimize the number of modifications I've made
# to the original HResNet code.


def centered_transform_mat(center_xy, rot_deg, scale, out_wh):
    half_width = out_wh[0] / 2.0
    half_height = out_wh[1] / 2.0
    translate_mat = np.float32([
        [1.0, 0.0, -center_xy[0] + half_width],
        [0.0, 1.0, -center_xy[1] + half_height],
        [0.0, 0.0, 1.0],
    ])

    rot_rad = rot_deg * np.pi / 180
    alpha = scale * np.cos(rot_rad)
    beta = scale * np.sin(rot_rad)
    rot_scale_mat = np.float32([
        [alpha, beta, (1 - alpha) * half_width - beta * half_height],
        [-beta, alpha, beta * half_width + (1 - alpha) * half_height],
        [0.0, 0.0, 1.0],
    ])

    return rot_scale_mat @ translate_mat


def random_occlusion(img, max_occlusion_size, opacity):

    assert img.ndim == 2 or img.ndim == 3

    nchan = 0
    if img.ndim == 2:
        img_height, img_width = img.shape
    elif img.ndim == 3:
        nchan, img_height, img_width = img.shape

    occ_center_x = np.random.rand() * img_width
    occ_center_y = np.random.rand() * img_height

    if np.random.rand() < 0.5:
        occ_min_x = occ_center_x - max_occlusion_size / 2
        occ_min_y = occ_center_y - max_occlusion_size / 2

        random_points = (
            np.random.rand(1, np.random.randint(3, 7), 2) * max_occlusion_size
            + np.array([[[occ_min_x, occ_min_y]]])
        )
        random_points = random_points.astype(np.int32)

        mask = np.zeros([img_height, img_width], dtype=np.uint8)
        cv2.fillPoly(mask, random_points, 255)
    else:
        mask = np.zeros([img_height, img_width], dtype=np.uint8)
        cv2.ellipse(
            mask,
            (int(occ_center_x), int(occ_center_y)),
            (int(max_occlusion_size * np.random.rand() / 2), int(max_occlusion_size * np.random.rand() / 2)),
            np.random.randint(0, 359),
            0, 360, 255, -1)
    mask = mask.astype(np.bool)
    if img.ndim == 3:
        mask = np.stack([mask] * nchan)

    rand_shade = np.random.randint(0, 255)
    img_float = img.astype(np.float32)
    img[mask] = img_float[mask] * (1 - opacity) + rand_shade * opacity
