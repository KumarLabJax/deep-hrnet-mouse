import cv2
import itertools
import numpy as np
import os
import re
import skimage.io
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter
from torchvision.transforms.functional import to_pil_image, to_tensor
import xml.etree.ElementTree as ET

from utils.transforms import affine_transform
from utils.xform import centered_transform_mat, random_occlusion


def parse_point_labels(cvat_xml_path, label_attr_name):
    root = ET.parse(cvat_xml_path)
    for image_elem in root.findall('./image'):
        img_name = image_elem.attrib['name']
        points_elems = itertools.chain(
            (
                pl for pl in image_elem.findall('./points')
                if pl.attrib['label'] == label_attr_name
            ),
            (
                pl for pl in image_elem.findall('./polyline')
                if pl.attrib['label'] == label_attr_name
            ),
        )

        xy_strs = []
        for points_elem in points_elems:
            xy_strs += [
                xy_str.split(',')
                for xy_str in points_elem.attrib['points'].split(';')
            ]

        assert len(xy_strs) 

        point_xy = np.array(xy_strs, dtype=np.float)
        yield {
            'image_name': img_name,
            'point_xy': point_xy,
        }


def transform_points(xy_points, xform):
    # need a row of 1's for affine transform matrix mult
    xy_points_xform = np.concatenate([
        xy_points,
        np.ones([1, xy_points.shape[1]], dtype=xy_points.dtype)])
    xy_points_xform = xform @ xy_points_xform

    return xy_points_xform[:2, :]


def _read_image(image_path):
    data_numpy = skimage.io.imread(image_path, as_gray=True) * 255

    data_numpy = data_numpy.round().astype(np.uint8)
    data_numpy = data_numpy[..., np.newaxis]

    return data_numpy


class SimplePointDataset(Dataset):

    def __init__(self, cfg, image_dir, point_labels, is_train, transform=None):
        self.cfg = cfg
        self.image_dir = image_dir
        self.point_labels = point_labels
        self.is_train = is_train
        self.transform = transform

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.flip = cfg.DATASET.FLIP

        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
        self.sigma = cfg.MODEL.SIGMA
        self.target_type = cfg.MODEL.TARGET_TYPE
        self.model_extra = cfg.MODEL.EXTRA

    def __len__(self):
        return len(self.point_labels)

    def _gen_heatmap(self, point_xy):
        target = np.zeros(
            (1, self.heatmap_size[1], self.heatmap_size[0]),
            dtype=np.float32)

        # build target heatmap where each point is the center of a 2D gaussian
        if self.target_type == 'gaussian':
            tmp_size = self.sigma * 3

            # # TODO can we add sub-pixel precision here?
            for curr_xy in point_xy:
                    feat_stride = self.image_size / self.heatmap_size
                    mu_x = int(curr_xy[0] / feat_stride[0] + 0.5)
                    mu_y = int(curr_xy[1] / feat_stride[1] + 0.5)
                    # Check that any part of the gaussian is in-bounds
                    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                    if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                            or br[0] < 0 or br[1] < 0:
                        # If not, just return the image as is
                        continue

                    # # Generate gaussian
                    size = 2 * tmp_size + 1
                    x = np.arange(0, size, 1, np.float32)
                    y = x[:, np.newaxis]
                    x0 = y0 = size // 2
                    # The gaussian is not normalized, we want the center value to equal 1
                    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                    # Usable gaussian range
                    g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                    g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                    # Image range
                    img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                    img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                    target[0, img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]],
                        target[0, img_y[0]:img_y[1], img_x[0]:img_x[1]])

        # build target heatmap where each point is the center of a 2D exponential
        # decay function
        elif self.target_type == 'exp_decay':

            # for now we require image_size and heatmap_size to be the
            # same, but we can change code to allow different sizes
            # later if needed
            assert np.all(self.image_size == self.heatmap_size)
            
            img_width, img_height = self.image_size

            # Each heat patch will just be a small square to save on
            # compute but large enough to allow almost all decay. For
            # a lambda of 1 a distance of 3 should be sufficient: (e^-3 = ~0.05).
            # We just need to scale this by 1/exp_decay_lambda to
            # make it work for any lambda
            EXP_DECAY_PATCH_SIZE_FACTOR = 4
            exp_decay_lambda = self.model_extra['EXP_DECAY_LAMBDA']
            heat_patch_size = EXP_DECAY_PATCH_SIZE_FACTOR / exp_decay_lambda

            for curr_xy in point_xy:

                    mu_x = curr_xy[0]
                    mu_y = curr_xy[1]

                    start_x = int(max(np.floor(mu_x - heat_patch_size), 0))
                    start_y = int(max(np.floor(mu_y - heat_patch_size), 0))

                    stop_x = int(min(np.ceil(mu_x + heat_patch_size + 1), img_width))
                    stop_y = int(min(np.ceil(mu_y + heat_patch_size + 1), img_height))

                    if start_x < stop_x and start_y < stop_y:
                        patch_width = stop_x - start_x
                        patch_height = stop_y - start_y

                        x = np.arange(start_x, stop_x) - mu_x
                        y = np.arange(start_y, stop_y) - mu_y

                        x_mat = np.tile(x, patch_height).reshape(patch_height, patch_width)
                        y_mat = np.tile(y, patch_width).reshape(patch_width, patch_height).T

                        xy_mat = np.stack([x_mat, y_mat], axis=2)
                        dist_mat = np.linalg.norm(xy_mat, axis=2)
                        decay_mat = np.exp(-exp_decay_lambda * dist_mat)

                        # we apply our 2D exponential decay patch to the target heatmap
                        # but we do it using maximum so that we get the desired result
                        # for overlapping patches
                        target[0, start_y:stop_y, start_x:stop_x] = np.maximum(
                            decay_mat,
                            target[0, start_y:stop_y, start_x:stop_x])

        # build target heatmap where each point is a single pixel set to 1.0
        elif self.target_type == 'point':

            # for now we require image_size and heatmap_size to be the
            # same, but we can change code to allow different sizes
            # later if needed
            assert np.all(self.image_size == self.heatmap_size)
            
            img_width, img_height = self.image_size

            for curr_xy in point_xy:

                    mu_x = int(round(curr_xy[0]))
                    mu_y = int(round(curr_xy[1]))

                    # print(type(joint_id), type(mu_y), type(mu_x))
                    # print(joint_id, mu_y, mu_x)
                    if 0 <= mu_x < img_width and 0 <= mu_y < img_height:
                        target[0, mu_y, mu_x] = 1.0

        # if we reach this else we've been given a target type that we don't
        # know how to deal with
        else:
            raise Exception('unexpected target type: {}'.format(self.target_type))

        return torch.tensor(target, dtype=torch.float32)

    def __getitem__(self, idx):

        point_label = self.point_labels[idx]
        image_name = point_label['image_name']
        point_xy = point_label['point_xy'].copy()

        image_size = np.array(self.cfg.MODEL.IMAGE_SIZE, dtype=np.uint32)

        image_path = os.path.join(self.image_dir, image_name)
        data_numpy = _read_image(image_path)

        # pick a random point between the min and max points for
        # the center_xy
        min_xy = point_xy.min(0)
        max_xy = point_xy.max(0)
        diff_xy = max_xy - min_xy
        center_xy = min_xy + diff_xy * np.random.rand(2)

        scale = self.cfg.DATASET.SCALE
        rot_deg = 0

        if self.is_train:
            sf = self.cfg.DATASET.SCALE_FACTOR
            scale *= np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            rot_deg = 360 * np.random.random() if np.random.random() <= 0.8 else 0

            if self.cfg.DATASET.FLIP and np.random.random() <= 0.5:
                # reflect the pixels along the X axis
                data_numpy = data_numpy[:, ::-1, ...]

                # center X needs to be adjusted along with all of the Xs in point_xy
                center_xy[0] = (data_numpy.shape[1] - 1) - center_xy[0]
                point_xy[:, 0] = (data_numpy.shape[1] - 1) - point_xy[:, 0]

        trans = centered_transform_mat(center_xy, rot_deg, scale, image_size)
        img = cv2.warpAffine(
            data_numpy,
            trans[:2, :],
            (image_size[0], image_size[1]),
            flags=cv2.INTER_LINEAR)

        # for training data we throw in some image augmentation:
        # brightness, contrast
        if self.is_train:
            jitter_brightness = self.cfg.DATASET.JITTER_BRIGHTNESS
            jitter_contrast = self.cfg.DATASET.JITTER_CONTRAST
            if jitter_brightness > 0 or jitter_contrast > 0:
                img = to_pil_image(img)
                img = ColorJitter(jitter_brightness, jitter_contrast)(img)
                img = to_tensor(img).numpy()
                img = (img * 255).astype(np.uint8)

            prob_randomized_occlusion = self.cfg.DATASET.PROB_RANDOMIZED_OCCLUSION
            max_occlusion_size = self.cfg.DATASET.MAX_OCCLUSION_SIZE
            occlusion_opacities = self.cfg.DATASET.OCCLUSION_OPACITIES
            if prob_randomized_occlusion > 0 and np.random.random() <= prob_randomized_occlusion:
                random_occlusion(img, max_occlusion_size, np.random.choice(occlusion_opacities))
        else:
            if img.ndim == 3:
                img = np.stack([img[:, :, i] for i in range(img.shape[2])])

        img = torch.from_numpy(img).to(torch.float32) / 255
        if img.dim() == 2:
            img = img.unsqueeze(0)

        # if we were provided an image augmentation in the constructor we use it here
        if self.transform:
            img = self.transform(img)

        point_xy = transform_points(point_xy.transpose(), trans).transpose()
        heatmap = self._gen_heatmap(point_xy)

        return {
            'image': img,
            'heatmap': heatmap,
        }
