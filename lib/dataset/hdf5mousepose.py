from collections import OrderedDict
import copy
import cv2
import logging
import h5py
import numpy as np
import random
import torch
from torchvision.transforms import ColorJitter
from torchvision.transforms.functional import to_pil_image, to_tensor

from dataset.JointsDataset import JointsDataset
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints
from utils.xform import centered_transform_mat, random_occlusion


NOSE_INDEX = 0

LEFT_EAR_INDEX = 1
RIGHT_EAR_INDEX = 2

BASE_NECK_INDEX = 3

LEFT_FRONT_PAW_INDEX = 4
RIGHT_FRONT_PAW_INDEX = 5

CENTER_SPINE_INDEX = 6

LEFT_REAR_PAW_INDEX = 7
RIGHT_REAR_PAW_INDEX = 8

BASE_TAIL_INDEX = 9
MID_TAIL_INDEX = 10
TIP_TAIL_INDEX = 11

logger = logging.getLogger(__name__)


class HDF5MousePose(JointsDataset):

    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)

        self.prob_randomized_occlusion = cfg.DATASET.PROB_RANDOMIZED_OCCLUSION
        self.max_occlusion_size = cfg.DATASET.MAX_OCCLUSION_SIZE
        self.occlusion_opacities = cfg.DATASET.OCCLUSION_OPACITIES
        self.prob_randomized_center = cfg.DATASET.PROB_RANDOMIZED_CENTER
        self.jitter_center = cfg.DATASET.JITTER_CENTER
        self.jitter_brightness = cfg.DATASET.JITTER_BRIGHTNESS
        self.jitter_contrast = cfg.DATASET.JITTER_CONTRAST
        self.jitter_saturation = cfg.DATASET.JITTER_SATURATION

        self.num_joints = 12

        self.flip_pairs = [
            [LEFT_EAR_INDEX, RIGHT_EAR_INDEX],
            [LEFT_FRONT_PAW_INDEX, RIGHT_FRONT_PAW_INDEX],
            [LEFT_REAR_PAW_INDEX, RIGHT_REAR_PAW_INDEX],
        ]

        self.db = self._get_db()

    def _get_db(self):

        def gen_db():
            with h5py.File(self.root, 'r') as hdf5file:
                if self.image_set in hdf5file:
                    for name, group in hdf5file[self.image_set].items():
                        if 'frames' in group and 'points' in group:
                            points = group['points']
                            for grp_frame_index in range(points.shape[0]):
                                grp_frame_pts = points[grp_frame_index, ...]
                                max_x, max_y = np.amax(grp_frame_pts, axis=0)
                                min_x, min_y = np.amin(grp_frame_pts, axis=0)

                                # width = max_x - min_x
                                # height = max_y - min_y

                                center_x = (max_x + min_x) / 2
                                center_y = (max_y + min_y) / 2
                                center_xy = np.array([center_x, center_y], dtype=np.float32)
                                # scale = np.array(
                                #     [
                                #         width * 1.0 / self.pixel_std,
                                #         height * 1.0 / self.pixel_std,
                                #     ],
                                #     dtype=np.float32)
                                # scale_range = 0.4
                                # scale = 1 + np.random.random([2]) * scale_range - scale_range / 2
                                scale = np.ones([2], dtype=np.float32)

                                joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
                                joints_3d[:, :2] = grp_frame_pts

                                yield {
                                    'image': (name, grp_frame_index),
                                    'center': center_xy,
                                    'scale': scale,
                                    'joints_3d': joints_3d,
                                    'joints_3d_vis': np.ones((self.num_joints, 3), dtype=np.float),
                                }

        return list(gen_db())

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''
        img_grp_name, img_grp_frame_index = db_rec['image']
        data_numpy = None
        with h5py.File(self.root, 'r') as hdf5file:
            data_numpy = hdf5file[self.image_set][img_grp_name]['frames'][img_grp_frame_index, ...]

        if data_numpy is None:
            logger.error('=> fail to read {}'.format(db_rec['image']))
            raise ValueError('Fail to read {}'.format(db_rec['image']))

        joints = db_rec['joints_3d']
        joints_vis = db_rec['joints_3d_vis']

        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        if self.is_train:
            sf = self.scale_factor
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            r = 360 * random.random() if random.random() <= 0.8 else 0

            if self.prob_randomized_center > 0 and random.random() <= self.prob_randomized_center:
                c[0] = data_numpy.shape[1] * random.random()
                c[1] = data_numpy.shape[0] * random.random()
            elif self.jitter_center > 0:
                c[0] += self.image_size[0] * self.jitter_center * np.random.randn()
                c[1] += self.image_size[1] * self.jitter_center * np.random.randn()

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                c[0] = data_numpy.shape[1] - c[0] - 1

        trans = centered_transform_mat(c, r, s[0], self.image_size)
        input = cv2.warpAffine(
            data_numpy,
            trans[:2, :],
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        if self.is_train:
            if self.jitter_brightness > 0 or self.jitter_contrast > 0 or self.jitter_saturation > 0:
                input = to_pil_image(input)
                input = ColorJitter(self.jitter_brightness, self.jitter_contrast, self.jitter_saturation)(input)
                input = to_tensor(input).squeeze(0).numpy()
                input = (input * 255).astype(np.uint8)

            if self.prob_randomized_occlusion > 0 and random.random() <= self.prob_randomized_occlusion:
                random_occlusion(input, self.max_occlusion_size, np.random.choice(self.occlusion_opacities))

        if self.transform:
            input = self.transform(input)

        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        target, target_weight = self.generate_target(joints, joints_vis)

        if not torch.is_tensor(target):
            target = torch.from_numpy(target)
        if not torch.is_tensor(target_weight):
            target_weight = torch.from_numpy(target_weight)

        meta = {
            'image': db_rec['image'],
            'filename': filename,
            'imgnum': imgnum,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score
        }

        input = torch.stack([input.squeeze(0)] * 3)

        return input, target, target_weight, meta
