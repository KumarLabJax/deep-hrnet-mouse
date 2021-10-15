import argparse
import colorsys
import itertools
import matplotlib.pyplot as plt
import numpy as np
import random
import skimage.draw as skidraw

import torchvision.transforms as transforms

import _init_paths
from config import cfg
from config import update_config

from dataset.multimousepose import MultiPoseDataset, parse_poses
import models


# Example:
#
#  python -u tools/testmultimousedata.py \
#       --cfg /home/sheppk/projects/deep-high-resolution-net.pytorch/experiments/multimouse/multimouse-1.yaml
#       --cvat-files \
#           /run/user/1002/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar/kumarlab-new/Brian/NeuralNets/MultiMousePose/Annotations/*.xml \
#           /run/user/1002/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar/kumarlab-new/Brian/NeuralNets/MultiMousePose/Annotations_NoMarkings/*.xml \
#       --image-dir '/run/user/1002/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar/kumarlab-new/Brian/NeuralNets/MultiMousePose/Dataset'

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = [colorsys.hsv_to_rgb(*c) for c in hsv]
    #random.shuffle(colors)
    return colors


def main():
    parser = argparse.ArgumentParser(description='test the multimouse pose dataset')

    parser.add_argument('--cvat-files',
                        help='list of CVAT XML files to use',
                        nargs='+',
                        required=True,
                        type=str)
    parser.add_argument('--image-dir',
                        help='directory containing images',
                        required=True,
                        type=str)
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()

    update_config(cfg, args)

    normalize = transforms.Normalize(
        mean=[0.485], std=[0.229]
    )

    all_poses = list(itertools.chain.from_iterable(parse_poses(f) for f in args.cvat_files))
    mpose_ds = MultiPoseDataset(
        cfg,
        args.image_dir,
        all_poses,
        True,
        normalize,
    )

    colors = random_colors(10)

    for _ in range(100):
        i = random.randrange(len(mpose_ds))
        print('doing', i)
        item = mpose_ds[i]

        print("item['image'].shape:", item['image'].shape)

        image = item['image'].numpy()

        chan_count = image.shape[0]
        plt_rows = 1 if chan_count == 1 else 2
        plt_cols = 3
        fig = plt.figure(figsize=(8, 8))

        for chan_index in range(chan_count):
            fig.add_subplot(plt_rows, plt_cols, chan_index + 1)
            plt.imshow(image[chan_index, ...], cmap='gray')

        fig.add_subplot(plt_rows, plt_cols, chan_count + 1)
        plt.imshow(item['joint_heatmaps'].numpy().max(0))

        pose_instances = item['pose_instances'][:item['instance_count'], ...]
        inst_image = np.zeros([image.shape[1], image.shape[2], 3], dtype=np.float32)
        inst_image_counts = np.zeros([image.shape[1], image.shape[2]], dtype=np.uint8)
        for instance_index, pose_instance in enumerate(pose_instances):
            for xy_point in pose_instance:
                temp_inst_image = np.zeros([image.shape[1], image.shape[2], 3], dtype=np.float32)
                rr, cc = skidraw.circle(xy_point[1], xy_point[0], 10, inst_image.shape)
                skidraw.set_color(temp_inst_image, (rr, cc), colors[instance_index])
                inst_image_counts[rr, cc] += 1
                inst_image += temp_inst_image
        inst_image /= np.expand_dims(inst_image_counts, 2)

        fig.add_subplot(plt_rows, plt_cols, chan_count + 2)
        plt.imshow(inst_image * np.expand_dims(item['joint_heatmaps'].numpy().max(0), 2))
        plt.show()


if __name__ == "__main__":
    main()
