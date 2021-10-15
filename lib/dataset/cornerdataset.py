# NOTE:
#   This code is based on hdf5mousepose.py code. It was named
#   hdf5mousepose.py by Massimo. I (KSS) renamed it so that the original
#   hdf5mousepose code will work as is but that means that this code will
#   not work until it is reintegrated back into the codebase

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

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class CornerDataset(JointsDataset):

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

        self.heatmap = np.array(cfg.MODEL.HEATMAP_SIZE)

        self.num_joints = 1
        # Changed this to 1

        self.db = self._get_db()

    def _get_db(self):

        def gen_db():
            with h5py.File(self.root, 'r') as hdf5file:
                if self.image_set in hdf5file:
                    for name, group in hdf5file[self.image_set].items():
                        if 'frames' in group and 'points' in group:
                            points = group['points']
                            # center_x = ((points[0, 0, 0] + points[0, 0, 1]) / 2)
                            # center_y = ((points[0, 1, 0] + points[0, 1, 1]) / 2)
                            # center = np.array([center_x, center_y], dtype=np.float32)

                            for i in range(4):

                                yield {
                                    'image_name': name,
                                    'object_index': i,
                                    'center': np.array(points[0, i, :], dtype=np.float32)
                                    }

                            # for grp_frame_index in range(points.shape[0]):
                            #     grp_frame_pts = points[grp_frame_index, ...]
                            #     max_x, max_y = np.amax(grp_frame_pts, axis=0)
                            #     min_x, min_y = np.amin(grp_frame_pts, axis=0)

                                # width = max_x - min_x
                                # height = max_y - min_y

                                # center_x = (max_x + min_x) / 2
                                # center_y = (max_y + min_y) / 2
                                # center_xy = np.array([center_x, center_y], dtype=np.float32)
                                # scale = np.array(
                                #     [
                                #         width * 1.0 / self.pixel_std,
                                #         height * 1.0 / self.pixel_std,
                                #     ],
                                #     dtype=np.float32)
                                # scale_range = 0.4
                                # scale = 1 + np.random.random([2]) * scale_range - scale_range / 2
                                # scale = np.ones([2], dtype=np.float32)

                                # joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
                                # joints_3d[:, :2] = grp_frame_pts

        return list(gen_db())
    #
    # def _gen_obj_indexes(self):
    #     for img_index, curr_obj in enumerate(self.object_labels):
    #         for obj_index in range(4):
    #             yield {
    #                 'image_index': img_index,
    #                 'object_index': obj_index,
    #             }

    def __len__(self):
        return len(self.db)

    def generate_target(self, center, joints_vis):
        '''
        :param center:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''

        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[0] = joints_vis[0]

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3

        feat_stride = self.image_size / self.heatmap_size
        mu_x = int(center[0] / feat_stride[0] + 0.5)
        mu_y = int(center[1] / feat_stride[1] + 0.5)
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                or br[0] < 0 or br[1] < 0:
            # If not, just return the image as is
            target_weight = 0

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

        v = target_weight
        if v > 0.5:
            target[img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''
        img_grp_name = db_rec['image_name']
        data_numpy = None
        with h5py.File(self.root, 'r') as hdf5file:
            data_numpy = hdf5file[self.image_set][img_grp_name]['frames'][0, ...]

        if data_numpy is None:
            logger.error('=> fail to read {}'.format(db_rec['image_name']))
            raise ValueError('Fail to read {}'.format(db_rec['image_name']))

        # joints = db_rec['joints_3d']
        # joints_vis = db_rec['joints_3d_vis']

        corner_loc = db_rec['center'].copy()
        cam_loc = db_rec['center'].copy()

        # s = db_rec['scale']
        # score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        # JITTER CENTER NEW
        # a = np.random.randint(0, 100)
        # if a > 50:
        cam_loc[0] += np.random.uniform(-50, 50)

        # b = np.random.randint(0, 100)
        # if b > 50:
        cam_loc[1] += np.random.uniform(-50, 50)

        if self.is_train:
            sf = self.scale_factor
            # s *= np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            r = 360 * random.random() if random.random() <= 0.8 else 0

            # if self.prob_randomized_center > 0 and random.random() <= self.prob_randomized_center:
            #     c[0] = data_numpy.shape[1] * random.random()
            #     c[1] = data_numpy.shape[0] * random.random()

        trans = centered_transform_mat(cam_loc, r, 1, self.image_size)
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

        joints_vis = np.ones((self.num_joints, 1), dtype=np.float32)

        if np.all(joints_vis > 0.0):
            corner_loc[0:2] = affine_transform(corner_loc[0:2], trans)

        # target_weight = np.ones((1, self.heatmap[1], self.heatmap[0]), dtype=np.float32)

        # joints_vis = np.ones((self.num_joints, 3), dtype=np.float32)
        # joints = np.zeros((self.num_joints, 3), dtype=np.float32)

        target, target_weight = self.generate_target(corner_loc, joints_vis)
        # self.generate_target(joints, joints_vis)
        # ???? how to gen weights?

        # if not torch.is_tensor(target):
        #    target = torch.from_numpy(target)
        # if not torch.is_tensor(target_weight):
        #    target_weight = torch.from_numpy(target_weight)

        meta = {
            'image': db_rec['image_name'],
            'filename': filename,
            'imgnum': imgnum,
            # 'joints': joints,
            # 'joints_vis': joints_vis,
            'center': cam_loc,
            # 'scale': s,
            'rotation': r,
            # 'score': score
        }

        input = torch.stack([input.squeeze(0)] * 3)

        # plt.imshow(input[0])
        # print('target.shape:', target.shape, target.dtype)
        # plt.imshow(target)
        # plt.show()
        _, axs = plt.subplots(1, 4, figsize=(12, 6))
        axs[0].imshow(input[0], aspect='equal')
        axs[1].imshow(input[1], aspect='equal')
        axs[2].imshow(input[2], aspect='equal')
        axs[3].imshow(target, aspect='equal')
        plt.show()

        print('input min max mean:', input.min(), input.max(), input.mean())
        print('target min max mean:', target.min(), target.max(), target.mean())

        return input, target, target_weight, meta
