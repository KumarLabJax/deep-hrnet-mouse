import argparse
import imageio
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import skimage
import yaml

import torch
import torch.backends.cudnn as cudnn

import _init_paths
from config import cfg
from config import update_config

from dataset.fecalbolidata import parse_fecal_boli_labels
from inferfecalbolicount import infer_fecal_boli_xy

import models


# Examples:
#
#   python -u tools/testfecalboli.py \
#       --model-file output-fecal-boli/fecalboli/pose_hrnet/fecalboli_2020-06-19_02/best_state.pth \
#       --cfg experiments/fecalboli/fecalboli_2020-06-19_02.yaml \
#       --cvat-files data/fecal-boli/*.xml \
#       --image-dir data/fecal-boli/images \
#       --image-list data/fecal-boli/fecal-boli-val-set.txt
#
#   python -u tools/testfecalboli.py \
#       --model-file output-fecal-boli/fecalboli/pose_hrnet/fecalboli_2020-06-19_02/best_state.pth \
#       --cfg experiments/fecalboli/fecalboli_2020-06-19_02.yaml \
#       --cvat-files data/fecal-boli/*.xml \
#       --image-dir data/fecal-boli/images \
#       --image-list data/fecal-boli/fecal-boli-val-set.txt \
#       --min-heatmap-val 0.3
#
#   python -u tools/testfecalboli.py \
#       --model-file output-fecal-boli/fecalboli/pose_hrnet/fecalboli_2020-06-19_02/best_state.pth \
#       --cfg experiments/fecalboli/fecalboli_2020-06-19_02.yaml \
#       --cvat-files data/fecal-boli/*.xml \
#       --image-dir data/fecal-boli/images \
#       --image-list data/fecal-boli/fecal-boli-val-set.txt \
#       --min-heatmap-val 0.3 \
#       --image-out-dir temp11


def gen_valid_point_combos(lbl_xy_list, inf_xy_list, max_dist):

    for lbl_xy in lbl_xy_list:
        for inf_xy in inf_xy_list:

            dist = np.linalg.norm(lbl_xy - inf_xy)
            if dist <= max_dist:
                yield {
                    'lbl_xy': lbl_xy,
                    'inf_xy': inf_xy,
                    'lbl_xy_tuple': tuple(lbl_xy),
                    'inf_xy_tuple': tuple(inf_xy),
                    'dist': dist,
                }


def render_overlays(raw_image, image_out_file, true_pos_xys, false_pos_xys, false_neg_xys):

    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca()

    for curr_xy in true_pos_xys:
        ax.add_artist(plt.Circle(curr_xy, 10, color='g', fill=False))
    for curr_xy in false_pos_xys:
        ax.add_artist(plt.Circle(curr_xy, 10, color='r', fill=False))
    for curr_xy in false_neg_xys:
        ax.add_artist(plt.Circle(curr_xy, 10, color='y', fill=False))

    plt.imshow(raw_image)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(image_out_file)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model-file',
        help='the model file to use for inference',
    )
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
        '--cfg',
        help='the configuration for the model to use for inference',
    )
    parser.add_argument(
        '--min-heatmap-val',
        type=float,
        default=0.75,
    )
    parser.add_argument(
        '--max-dist-px',
        type=float,
        default=5.0,
    )
    parser.add_argument(
        '--image-out-dir',
        type=str,
    )

    args = parser.parse_args()

    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.TEST.MODEL_FILE = args.model_file
    cfg.freeze()

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    if args.image_out_dir:
        os.makedirs(args.image_out_dir, exist_ok=True)

    with torch.no_grad():

        model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
            cfg, is_train=False
        )
        # print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
        model.eval()
        model = model.cuda()

        # normalize = transforms.Normalize(
        #     mean=[0.485], std=[0.229]
        # )

        image_list_filename = args.image_list
        img_names = None
        if image_list_filename is not None:
            img_names = set()
            with open(image_list_filename) as val_file:
                for curr_line in val_file:
                    img_name = curr_line.strip()
                    img_names.add(img_name)

        fecal_boli_labels = list(itertools.chain.from_iterable(
            parse_fecal_boli_labels(f) for f in args.cvat_files))
        if img_names is not None:
            fecal_boli_labels = [
                lbl for lbl in fecal_boli_labels
                if lbl['image_name'] in img_names
            ]

        accuracies = []
        avg_pixel_errors = []
        precisions = []
        recalls = []

        print('\t'.join(['Name', 'Accuracy', 'Average Pixel Error', 'Precision', 'Recall']))
        for lbl in fecal_boli_labels:
            
            image_path = os.path.join(args.image_dir, lbl['image_name'])
            image_data_numpy = skimage.io.imread(image_path, as_gray=False)

            inf_xy_vals = infer_fecal_boli_xy(
                model,
                [image_data_numpy],
                args.min_heatmap_val,
            )

            inf_xy_list = list(next(inf_xy_vals).numpy())
            lbl_xy_list = list(lbl['fecal_boli_xy'])

            point_combos = gen_valid_point_combos(lbl_xy_list, inf_xy_list, args.max_dist_px)
            point_combos = sorted(point_combos, key=lambda pc: pc['dist'])

            labels_found = set()
            infs_found = set()
            best_point_combos = []
            for pc in point_combos:
                if pc['lbl_xy_tuple'] not in labels_found and pc['inf_xy_tuple'] not in infs_found:
                    labels_found.add(pc['lbl_xy_tuple'])
                    infs_found.add(pc['inf_xy_tuple'])
                    best_point_combos.append(pc)

            true_pos = len(best_point_combos)
            false_neg = len(lbl_xy_list) - true_pos
            false_pos = len(inf_xy_list) - true_pos

            acc = true_pos / (true_pos + false_neg + false_pos)
            avg_pixel_err = np.mean([pc['dist'] for pc in best_point_combos])
            precision = true_pos / (true_pos + false_pos)
            recall = true_pos / (true_pos + false_neg)

            accuracies.append(acc)
            avg_pixel_errors.append(avg_pixel_err)
            precisions.append(precision)
            recalls.append(recall)

            if args.image_out_dir:
                true_pos_xys = [pc['inf_xy'] for pc in best_point_combos]
                false_pos_xys = [(x, y) for x, y in inf_xy_list if (x, y) not in infs_found]
                false_neg_xys = [(x, y) for x, y in lbl_xy_list if (x, y) not in labels_found]
                image_name_root, image_name_ext = os.path.splitext(lbl['image_name'])
                image_out_file = os.path.join(args.image_out_dir, image_name_root + '_fb_validation.png')
                render_overlays(
                    image_data_numpy,
                    image_out_file,
                    true_pos_xys,
                    false_pos_xys,
                    false_neg_xys,
                )

            print('\t'.join([
                lbl['image_name'],
                str(acc),
                str(avg_pixel_err),
                str(precision),
                str(recall),
            ]))

        print('\t'.join([
            'total avg',
            str(np.mean(accuracies)),
            str(np.mean(avg_pixel_errors)),
            str(np.mean(precisions)),
            str(np.mean(recalls)),
        ]))

if __name__ == "__main__":
    main()
