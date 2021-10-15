import argparse
import colorsys
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import scipy.stats
import skimage.draw
import skimage.io
import torch
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms

import _init_paths
import utils.assocembedutil as aeutil
from config import cfg
from config import update_config
from infercorners import argmax_2d

#from dataset.multimousepose import MultiPoseDataset, parse_poses, decompose_frame_name
from dataset.simplepointdata import parse_point_labels
import models


CM_PER_PIXEL = 19.5 * 2.54 / 400


# Example use:
# python -u tools/testcornermodel.py \
#       --model-file output-corner/simplepoint/pose_hrnet/corner_2020-06-30_01/best_state.pth \
#       --cfg experiments/corner/corner_2020-06-30_01.yaml \
#       --cvat-files data/corner/*.xml \
#       --image-dir data/corner/corner-images \
#       --image-list data/corner/corner-val-set-LL-only.txt

def main():

    parser = argparse.ArgumentParser(description='test the corner model')

    parser.add_argument(
        '--cvat-files',
        help='list of CVAT XML files to use',
        nargs='+',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--image-dir',
        help='directory containing images',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--image-list',
        help='file containing newline separated list of images to use',
        default=None,
    )
    parser.add_argument(
        '--model-file',
        help='the model file to use for inference',
        required=True,
    )
    parser.add_argument(
        '--cfg',
        help='the configuration for the model to use for inference',
        required=True,
        type=str,
    )

    args = parser.parse_args()

    print('=> loading configuration from {}'.format(args.cfg))

    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    if args.model_file:
        cfg.TEST.MODEL_FILE = args.model_file
    cfg.freeze()

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    with torch.no_grad():

        model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
            cfg, is_train=False
        )
        print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
        model.eval()
        model = model.cuda()

        normalize = transforms.Normalize(
            mean=[0.485], std=[0.229]
        )
        xform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.45, 0.45, 0.45],
                std=[0.225, 0.225, 0.225],
            ),
        ])

        image_list_filename = args.image_list
        img_names = None
        if image_list_filename is not None:
            img_names = set()
            with open(image_list_filename) as val_file:
                for curr_line in val_file:
                    img_name = curr_line.strip()
                    img_names.add(img_name)

        pose_labels = list(itertools.chain.from_iterable(
            parse_point_labels(f, 'corner') for f in args.cvat_files))
        if img_names is not None:
            pose_labels = [p for p in pose_labels if p['image_name'] in img_names]

        point_error_dists = []
        pose_dist_avg_sum = 0
        for pose_label in pose_labels:
            image_name = pose_label['image_name']
            # label_pose_instances = [
            #     aeutil.PoseInstance.from_xy_tensor(t)
            #     for t in pose_label['pose_instances']
            # ]
            print('=============================')
            print('image_name:', image_name)
            print('== LABELS ==')
            print(pose_label['point_xy'])
            # print([pi.keypoints for pi in label_pose_instances])


            image_path = os.path.join(args.image_dir, image_name)

            #image_data_numpy = skimage.io.imread(image_path, as_gray=True)

            #image_data = torch.from_numpy(image_data_numpy).to(torch.float32)

            image_data_numpy = skimage.io.imread(image_path)
            image_data = xform(image_data_numpy)
            # image_data = normalize(image_data.unsqueeze(0)).squeeze(0)
            #image_data = torch.stack([image_data] * 3)

            # add a size 1 batch dimension to the image and move it to the GPU
            batch_tensor = image_data.unsqueeze(0).cuda()

            x = model(batch_tensor)

            x.squeeze_(-3)

            img_h = batch_tensor.size(-2)
            img_w = batch_tensor.size(-1)

            x_ul = x[:, :(img_h // 2), :(img_w // 2)]
            x_ll = x[:, (img_h // 2):, :(img_w // 2)]
            x_ur = x[:, :(img_h // 2), (img_w // 2):]
            x_lr = x[:, (img_h // 2):, (img_w // 2):]

            maxvals_ul, preds_ul = argmax_2d(x_ul)
            maxvals_ll, preds_ll = argmax_2d(x_ll)
            maxvals_ur, preds_ur = argmax_2d(x_ur)
            maxvals_lr, preds_lr = argmax_2d(x_lr)

            preds_ul = preds_ul.cpu().numpy().astype(np.uint16)
            preds_ll = preds_ll.cpu().numpy().astype(np.uint16)
            preds_ur = preds_ur.cpu().numpy().astype(np.uint16)
            preds_lr = preds_lr.cpu().numpy().astype(np.uint16)

            preds_ll[..., 0] += img_h // 2
            preds_ur[..., 1] += img_w // 2
            preds_lr[..., 0] += img_h // 2
            preds_lr[..., 1] += img_w // 2

            pred_stack = np.stack([preds_ul, preds_ll, preds_ur, preds_lr], axis=-2)
            pred_stack = pred_stack[..., [-1, -2]] # go from (y, x) to (x, y)
            pred_stack = np.squeeze(pred_stack, axis=0)
            print('== INFERENCE ==')
            print(pred_stack)
            print()

            _, axs = plt.subplots(1, 2, figsize=(12, 6))

            axs[0].imshow(skimage.io.imread(image_path, as_gray=True))

            max_heatmap_np = x[0, ...].cpu().numpy()
            # max_heatmap_np[20, 20] = 1
            axs[1].imshow(max_heatmap_np)

            image_base, _ = os.path.splitext(os.path.basename(image_name))
            plt.savefig(os.path.join(
                'temp',
                'corner',
                image_base + '_corner_heatmap.png'))

            plt.close()

            max_dist = np.linalg.norm(np.array([img_h / 2, img_w / 2]))
            for i in range(4):
                curr_best_dist = max_dist
                curr_pred = pred_stack[i, :]
                for j in range(4):
                    curr_lbl = pose_label['point_xy'][j, :]
                    curr_dist = np.linalg.norm(curr_pred - curr_lbl)
                    if curr_dist < curr_best_dist:
                        curr_best_dist = curr_dist
                point_error_dists.append(curr_best_dist)

            print(point_error_dists[-4:])

        pixel_err_dist_sem = scipy.stats.sem(point_error_dists, axis=None, nan_policy='omit')
        pixel_err_dist_mean = np.mean(point_error_dists)
        print(f'Pixel MAE: {pixel_err_dist_mean:.2f} ±{pixel_err_dist_sem:.2f} {pixel_err_dist_mean * CM_PER_PIXEL:.2f} ±{pixel_err_dist_sem * CM_PER_PIXEL:.2f}')
        print(sorted(point_error_dists, reverse=True)[:10])


if __name__ == "__main__":
    main()
