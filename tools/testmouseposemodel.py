import argparse
import h5py
import numpy as np
import scipy.stats
import yaml

import matplotlib.pyplot as plt

import torch
import torch.nn.parallel
import torch.nn.functional as torchfunc
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from config import cfg
from core.inference import get_final_preds
from core.inference import get_max_preds

import dataset
import models

CM_PER_PIXEL = 19.5 * 2.54 / 400


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

INDEX_NAMES = [
    'Nose',

    'Left Ear',
    'Right Ear',

    'Base Neck',

    'Left Front Paw',
    'Right Front Paw',

    'Center Spine',

    'Left Rear Paw',
    'Right Rear Paw',

    'Base Tail',
    'Mid Tail',
    'Tip Tail',
]

# Examples:
#
#   python -u tools/testmouseposemodel.py \
#       --model-file ../pose-est-env/pose-est-model.pth \
#       ../pose-est-env/pose-est-conf.yaml
#
#   python -u tools/testmouseposemodel.py \
#       --model-file ../pose-est-env/pose-est-model.pth \
#       --category-yaml data/hdf5mouse/merged_pose_annos_mouse_categories_2019-06-26.yaml \
#       --category-count-cap 200 \
#       ../pose-est-env/pose-est-conf.yaml
#
#   python -u tools/testmouseposemodel.py \
#       --model-file ../pose-est-env/pose-est-model.pth \
#       --category-yaml data/hdf5mouse/diverse-strain-poses-categories.yaml \
#       --dataset-root data/hdf5mouse/diverse-strain-poses.h5 \
#       ../pose-est-env/pose-est-conf.yaml
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model-file',
        help='the model file to use for inference',
        default=None,
    )

    parser.add_argument(
        '--dataset-root',
        help='the dataset to use for inference',
        default=None,
    )

    parser.add_argument(
        '--category-yaml',
        help='a YAML file describing which category the validation images fall into',
        default=None,
    )

    parser.add_argument(
        '--category-count-cap',
        help='if this is selected we shuffle then cap the count in each category',
        type=int,
        required=False,
    )

    parser.add_argument(
        'cfg',
        help='the configuration for the model to use for inference',
    )

    args = parser.parse_args()

    print('=> loading configuration from {}'.format(args.cfg))

    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    if args.model_file:
        cfg.TEST.MODEL_FILE = args.model_file
    if args.dataset_root:
        cfg.DATASET.ROOT = args.dataset_root
    cfg.freeze()

    name_category_map = dict()
    if args.category_yaml:
        with open(args.category_yaml, 'r') as category_yaml_file:
            category_dict_list = yaml.safe_load(category_yaml_file)
            for category_dict in category_dict_list:
                for group_name in category_dict['group_names']:
                    # print(category_dict['category_name'], group_name)
                    name_category_map[group_name] = category_dict['category_name']

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
        cfg, is_train=False
    )
    print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
    model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    model.eval()
    model = model.cuda()

    xform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.45],
            std=[0.225],
        ),
    ])

    with torch.no_grad():

        category_pixel_err_dists = dict()
        with h5py.File(cfg.DATASET.ROOT, 'r') as hdf5file:

            for name, group in hdf5file[cfg.DATASET.TEST_SET].items():
                category = 'default'
                if name in name_category_map:
                    category = name_category_map[name]

                # print('NAME:', name, category)
                if category not in category_pixel_err_dists:
                    category_pixel_err_dists[category] = []
                
                if 'frames' in group and 'points' in group:
                    points = group['points']
                    for grp_frame_index in range(points.shape[0]):
                        grp_frame_pts = points[grp_frame_index, ...]

                        data_numpy = group['frames'][grp_frame_index, ...]
                        data_numpy = data_numpy.squeeze(2)
                        data = xform(data_numpy).squeeze(0)
                        data = data.cuda()
                        data = torch.stack([data] * 3)
                        data = data.unsqueeze(0)

                        # print(grp_frame_pts.shape)
                        # print(data.shape)

                        inf_out = model(data)
                        in_out_ratio = data.size(-1) // inf_out.size(-1)
                        if in_out_ratio == 4:
                            # print('need to upscale')
                            inf_out = torchfunc.upsample(inf_out, scale_factor=4, mode='bicubic', align_corners=False)
                        inf_out = inf_out.cpu().numpy()
                        # print('inf_out.shape:', inf_out.shape)

                        preds, maxvals = get_max_preds(inf_out)
                        preds = preds.astype(np.uint16).squeeze(0)
                        maxvals = maxvals.squeeze(2).squeeze(0)

                        pixel_err = preds.astype(np.float32) - grp_frame_pts
                        pixel_err_dist = np.linalg.norm(pixel_err, ord=2, axis=1)
                        category_pixel_err_dists[category].append(pixel_err_dist)

        rng = np.random.default_rng(1111)
        for category, pixel_err_dists in category_pixel_err_dists.items():

            if args.category_count_cap is not None:
                rng.shuffle(pixel_err_dists)
                pixel_err_dists = pixel_err_dists[:args.category_count_cap]

            print()
            print('=======================')
            print('DATA CATEGORY:', category, 'COUNT:', len(pixel_err_dists))

            pixel_err_dists = np.stack(pixel_err_dists)

            pixel_err_dist_mean = np.nanmean(pixel_err_dists)
            pixel_err_dist_sem = scipy.stats.sem(pixel_err_dists, axis=None, nan_policy='omit')

            pixel_err_dist_mean = np.nanmean(pixel_err_dists)
            pixel_err_dist_sem = scipy.stats.sem(pixel_err_dists, axis=None, nan_policy='omit')

            pixel_err_dist_means = np.nanmean(pixel_err_dists, axis=0)
            pixel_dist_sems = scipy.stats.sem(pixel_err_dists, axis=0, nan_policy='omit')

            print(pixel_err_dist_mean)
            print(pixel_err_dist_sem)
            print(f'OVERALL MAE: {pixel_err_dist_mean:.2f} ±{pixel_err_dist_sem:.2f} {pixel_err_dist_mean * CM_PER_PIXEL:.2f} ±{pixel_err_dist_sem * CM_PER_PIXEL:.2f}')
            print()

            print(f'NOSE Pixel MAE:              {pixel_err_dist_means[NOSE_INDEX]:.2f} ±{pixel_dist_sems[NOSE_INDEX]:.2f} {pixel_err_dist_means[NOSE_INDEX] * CM_PER_PIXEL:.2f} ±{pixel_dist_sems[NOSE_INDEX] * CM_PER_PIXEL:.2f}')
            print(f'LEFT_EAR Pixel MAE:          {pixel_err_dist_means[LEFT_EAR_INDEX]:.2f} ±{pixel_dist_sems[LEFT_EAR_INDEX]:.2f} {pixel_err_dist_means[LEFT_EAR_INDEX] * CM_PER_PIXEL:.2f} ±{pixel_dist_sems[LEFT_EAR_INDEX] * CM_PER_PIXEL:.2f}')
            print(f'RIGHT_EAR Pixel MAE:         {pixel_err_dist_means[RIGHT_EAR_INDEX]:.2f} ±{pixel_dist_sems[RIGHT_EAR_INDEX]:.2f} {pixel_err_dist_means[RIGHT_EAR_INDEX] * CM_PER_PIXEL:.2f} ±{pixel_dist_sems[RIGHT_EAR_INDEX] * CM_PER_PIXEL:.2f}')
            print(f'BASE_NECK Pixel MAE:         {pixel_err_dist_means[BASE_NECK_INDEX]:.2f} ±{pixel_dist_sems[BASE_NECK_INDEX]:.2f} {pixel_err_dist_means[BASE_NECK_INDEX] * CM_PER_PIXEL:.2f} ±{pixel_dist_sems[BASE_NECK_INDEX] * CM_PER_PIXEL:.2f}')
            print(f'LEFT_FRONT_PAW Pixel MAE:    {pixel_err_dist_means[LEFT_FRONT_PAW_INDEX]:.2f} ±{pixel_dist_sems[LEFT_FRONT_PAW_INDEX]:.2f} {pixel_err_dist_means[LEFT_FRONT_PAW_INDEX] * CM_PER_PIXEL:.2f} ±{pixel_dist_sems[LEFT_FRONT_PAW_INDEX] * CM_PER_PIXEL:.2f}')
            print(f'RIGHT_FRONT_PAW Pixel MAE:   {pixel_err_dist_means[RIGHT_FRONT_PAW_INDEX]:.2f} ±{pixel_dist_sems[RIGHT_FRONT_PAW_INDEX]:.2f} {pixel_err_dist_means[RIGHT_FRONT_PAW_INDEX] * CM_PER_PIXEL:.2f} ±{pixel_dist_sems[RIGHT_FRONT_PAW_INDEX] * CM_PER_PIXEL:.2f}')
            print(f'CENTER_SPINE Pixel MAE:      {pixel_err_dist_means[CENTER_SPINE_INDEX]:.2f} ±{pixel_dist_sems[CENTER_SPINE_INDEX]:.2f} {pixel_err_dist_means[CENTER_SPINE_INDEX] * CM_PER_PIXEL:.2f} ±{pixel_dist_sems[CENTER_SPINE_INDEX] * CM_PER_PIXEL:.2f}')
            print(f'LEFT_REAR_PAW Pixel MAE:     {pixel_err_dist_means[LEFT_REAR_PAW_INDEX]:.2f} ±{pixel_dist_sems[LEFT_REAR_PAW_INDEX]:.2f} {pixel_err_dist_means[LEFT_REAR_PAW_INDEX] * CM_PER_PIXEL:.2f} ±{pixel_dist_sems[LEFT_REAR_PAW_INDEX] * CM_PER_PIXEL:.2f}')
            print(f'RIGHT_REAR_PAW Pixel MAE:    {pixel_err_dist_means[RIGHT_REAR_PAW_INDEX]:.2f} ±{pixel_dist_sems[RIGHT_REAR_PAW_INDEX]:.2f} {pixel_err_dist_means[RIGHT_REAR_PAW_INDEX] * CM_PER_PIXEL:.2f} ±{pixel_dist_sems[RIGHT_REAR_PAW_INDEX] * CM_PER_PIXEL:.2f}')
            print(f'BASE_TAIL Pixel MAE:         {pixel_err_dist_means[BASE_TAIL_INDEX]:.2f} ±{pixel_dist_sems[BASE_TAIL_INDEX]:.2f} {pixel_err_dist_means[BASE_TAIL_INDEX] * CM_PER_PIXEL:.2f} ±{pixel_dist_sems[BASE_TAIL_INDEX] * CM_PER_PIXEL:.2f}')
            print(f'MID_TAIL Pixel MAE:          {pixel_err_dist_means[MID_TAIL_INDEX]:.2f} ±{pixel_dist_sems[MID_TAIL_INDEX]:.2f} {pixel_err_dist_means[MID_TAIL_INDEX] * CM_PER_PIXEL:.2f} ±{pixel_dist_sems[MID_TAIL_INDEX] * CM_PER_PIXEL:.2f}')
            print(f'TIP_TAIL Pixel MAE:          {pixel_err_dist_means[TIP_TAIL_INDEX]:.2f} ±{pixel_dist_sems[TIP_TAIL_INDEX]:.2f} {pixel_err_dist_means[TIP_TAIL_INDEX] * CM_PER_PIXEL:.2f} ±{pixel_dist_sems[TIP_TAIL_INDEX] * CM_PER_PIXEL:.2f}')
            print()


if __name__ == "__main__":
    main()
