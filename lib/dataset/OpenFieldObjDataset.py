import cv2
import numpy as np
import os
import random
import skimage.draw
import skimage.io
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter
from torchvision.transforms.functional import to_pil_image, to_tensor
import xml.etree.ElementTree as ET

from utils.xform import centered_transform_mat, random_occlusion


def parse_obj_labels(cvat_xml_path):
    root = ET.parse(cvat_xml_path)
    for image_elem in root.findall('./image'):
        img_name = image_elem.attrib['name']
        object_polygon_elems = (
            pl for pl in image_elem.findall('./polygon')
            if pl.attrib['label'] == 'object'
        )

        object_polygon_points = []
        for object_polygon_elem in object_polygon_elems:
            xy_strs = [
                xy_str.split(',')
                for xy_str in object_polygon_elem.attrib['points'].split(';')
            ]
            assert len(xy_strs) >= 3

            xy_points = np.array(
                [(float(x_str), float(y_str)) for x_str, y_str in xy_strs],
                dtype=np.float32,
            )
            xy_points = np.transpose(xy_points)

            object_polygon_points.append(xy_points)

        yield {
            'image_name': img_name,
            'object_polygons': object_polygon_points,
        }


def transform_points(xy_points, xform):
    # need a row of 1's for affine transform matrix mult
    xy_points_xform = np.concatenate([
        xy_points,
        np.ones([1, xy_points.shape[1]], dtype=xy_points.dtype)])
    xy_points_xform = xform @ xy_points_xform

    return xy_points_xform[:2, :]


class OpenFieldObjDataset(Dataset):

    def __init__(self, cfg, object_labels, is_train, transform=None):
        self.cfg = cfg
        self.object_labels = object_labels
        self.object_indexes = list(self._gen_obj_indexes())
        self.is_train = is_train
        self.transform = transform

    def _gen_obj_indexes(self):
        for img_index, curr_obj in enumerate(self.object_labels):
            for obj_index in range(len(curr_obj['object_polygons'])):
                yield {
                    'image_index': img_index,
                    'object_index': obj_index,
                }

    def __len__(self):
        return len(self.object_indexes)

    def __getitem__(self, idx):
        curr_obj_indexes = self.object_indexes[idx]
        image_index = curr_obj_indexes['image_index']
        object_index = curr_obj_indexes['object_index']

        image_size = np.array(self.cfg.MODEL.IMAGE_SIZE, dtype=np.uint32)

        labels = self.object_labels[image_index]
        image_name = labels['image_name']
        object_polygons = labels['object_polygons']
        selected_object_polygon = object_polygons[object_index]

        image_path = os.path.join(self.cfg.DATASET.ROOT, image_name)
        data_numpy = skimage.io.imread(image_path, as_gray=True) * 255
        data_numpy = data_numpy.round().astype(np.uint8)
        data_numpy = data_numpy[..., np.newaxis]

        center_xy = (selected_object_polygon.min(1) + selected_object_polygon.max(1)) / 2.0
        scale = 1.0
        rot_deg = 0

        if self.is_train:
            sf = self.cfg.DATASET.SCALE_FACTOR
            scale *= np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            rot_deg = 360 * random.random() if random.random() <= 0.8 else 0

            prob_randomized_center = self.cfg.DATASET.PROB_RANDOMIZED_CENTER
            jitter_center = self.cfg.DATASET.JITTER_CENTER
            if prob_randomized_center > 0 and random.random() <= prob_randomized_center:
                center_xy[0] = data_numpy.shape[0] * random.random()
                center_xy[1] = data_numpy.shape[1] * random.random()
            elif jitter_center > 0:
                center_xy[0] += image_size[1] * jitter_center * np.random.randn()
                center_xy[1] += image_size[0] * jitter_center * np.random.randn()

            if self.cfg.DATASET.FLIP and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                center_xy[0] = data_numpy.shape[0] - center_xy[0] - 1

                for obj_poly in object_polygons:
                    obj_poly[0, :] = data_numpy.shape[0] - obj_poly[0, :] - 1

        trans = centered_transform_mat(center_xy, rot_deg, scale, image_size)
        img = cv2.warpAffine(
            data_numpy,
            trans[:2, :],
            (image_size[0], image_size[1]),
            flags=cv2.INTER_LINEAR)

        if self.is_train:
            jitter_brightness = self.cfg.DATASET.JITTER_BRIGHTNESS
            jitter_contrast = self.cfg.DATASET.JITTER_CONTRAST
            jitter_saturation = self.cfg.DATASET.JITTER_SATURATION
            if jitter_brightness > 0 or jitter_contrast > 0 or jitter_saturation > 0:
                img = to_pil_image(img)
                img = ColorJitter(jitter_brightness, jitter_contrast, jitter_saturation)(img)
                img = to_tensor(img).squeeze(0).numpy()
                img = (img * 255).astype(np.uint8)

            prob_randomized_occlusion = self.cfg.DATASET.PROB_RANDOMIZED_OCCLUSION
            max_occlusion_size = self.cfg.DATASET.MAX_OCCLUSION_SIZE
            occlusion_opacities = self.cfg.DATASET.OCCLUSION_OPACITIES
            if prob_randomized_occlusion > 0 and random.random() <= prob_randomized_occlusion:
                random_occlusion(img, max_occlusion_size, np.random.choice(occlusion_opacities))

        if self.transform:
            img = self.transform(img)

        # image size is width, height which means we reverse the order for creating a numpy array
        seg_target = torch.zeros(image_size[1], image_size[0], dtype=torch.float32)
        for obj_poly in object_polygons:
            xformed_obj_poly = transform_points(obj_poly, trans)

            # scikit image expects row followed by column so we give y, x order
            rr, cc = skimage.draw.polygon(xformed_obj_poly[1, :], xformed_obj_poly[0, :])

            # mask out any out-of-bounds indexes
            rc_mask = (rr < image_size[1]) & (cc < image_size[0])
            rr = rr[rc_mask]
            cc = cc[rc_mask]

            seg_target[rr, cc] = 1.0

        # add channel dimension
        seg_target = seg_target.unsqueeze(0)

        return img, seg_target
