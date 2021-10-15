import argparse
import h5py
import imageio
import numpy as np
import time
import yaml

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
from config import update_config

import models
import cv2
import string
import os.path

import skimage.draw
import skimage.io

FRAMES_PER_MINUTE = 30 * 60


# Example use:
#
#   time python -u tools/infercorners.py \
#       --model-file output-full-mouse-pose/hdf5mousepose/pose_hrnet/corner-detection/model_best.pth \
#       --cfg corner-detection.yaml \
#       --root-dir ~/smb/labshare \
#       --batch-file netfiles.csv
#
#   time python -u tools/infercorners.py \
#       --model-file output-corner/simplepoint/pose_hrnet/corner_2020-06-30_01/best_state.pth \
#       --cfg experiments/corner/corner_2020-06-30_01.yaml \
#       --root-dir ~/smb/labshare \
#       --batch-file /home/sheppk/projects/massimo-deep-hres-net/netfiles.csv
#
#   time python -u tools/infercorners.py \
#       --model-file output-corner/simplepoint/pose_hrnet/corner_2020-06-30_01/best_state.pth \
#       --cfg experiments/corner/corner_2020-06-30_01.yaml \
#       --root-dir ~/smb/labshare \
#       --batch-file data/corner/leinani-corner-batch-2020-08-20.txt
#
#   share_root='/run/user/1000/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar'
#   time python -u tools/infercorners.py \
#       --model-file output-corner/simplepoint/pose_hrnet/corner_2020-06-30_01/best_state.pth \
#       --cfg experiments/corner/corner_2020-06-30_01.yaml \
#       --root-dir "${share_root}" \
#       --batch-file ~/projects/gaitanalysis/data/metadata/strain-survey-b6j-bjnj-only-batch-2021-01-18.txt

def argmax_2d(tensor):

    assert tensor.dim() >= 2
    max_col_vals, max_cols = torch.max(tensor, -1, keepdim=True)
    max_vals, max_rows = torch.max(max_col_vals, -2, keepdim=True)
    max_cols = torch.gather(max_cols, -2, max_rows)

    max_vals = max_vals.squeeze(-1).squeeze(-1)
    max_rows = max_rows.squeeze(-1).squeeze(-1)
    max_cols = max_cols.squeeze(-1).squeeze(-1)

    return max_vals, torch.stack([max_rows, max_cols], -1)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--cfg',
        required=True,
        help='the configuration for the model to use for inference',
    )
    parser.add_argument(
        '--model-file',
        required=True,
        help='the model file to use for inference',
    )
    parser.add_argument(
        '--batch-file',
        required=False,
        help='the batch file listing videos to process',
    )
    parser.add_argument(
        '--root-dir',
        required=False,
        help='the root directory that batch file paths are build off of'
    )
    parser.add_argument(
        '--videos',
        required=False,
        nargs='+',
        help='specify video paths on the command line as an alternative'
             ' to using the "--batch-file" and "--root-dir" arguments',
    )
    

    args = parser.parse_args()
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    if args.model_file:
        cfg.TEST.MODEL_FILE = args.model_file
    cfg.freeze()

    start_time = time.time()

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
            mean=[0.45, 0.45, 0.45],
            std=[0.225, 0.225, 0.225],
        ),
    ])

    video_filenames = []
    if args.batch_file:
        with open(args.batch_file) as batch_file:
            for line in batch_file:
                vid_filename = line.strip()
                if vid_filename:
                    video_filename = os.path.join(args.root_dir, vid_filename)
                    video_filenames.append(video_filename)

    if args.videos:
        for video_filename in args.videos:
            video_filenames.append(video_filename)

    with torch.no_grad():
        for video_filename in video_filenames:
            if not os.path.isfile(video_filename):
                print(f'ERROR: {video_filename} is not a valid file')
                continue

            with imageio.get_reader(video_filename) as reader:

                all_preds = []
                all_maxvals = []
                batch = []

                def perform_inference():
                    if batch:
                        batch_tensor = torch.stack([xform(img) for img in batch]).cuda()
                        batch.clear()

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

                        pred_stack = np.stack(
                            [preds_ul, preds_ll, preds_ur, preds_lr],
                            axis=-2,
                        )

                        all_preds.append(pred_stack)

                last_frame_index = 600
                frame_step_size = 100
                for frame_index, image in enumerate(reader):

                    if frame_index == 0:
                        mockup = image

                    if frame_index % frame_step_size == 0:

                        batch.append(image)
                        perform_inference()

                    if frame_index == last_frame_index:
                        break

                all_preds = np.concatenate(all_preds)

                xmed_ul = []
                xmed_ll = []
                xmed_ur = []
                xmed_lr = []

                ymed_ul = []
                ymed_ll = []
                ymed_ur = []
                ymed_lr = []

                for i in range(len(all_preds[0])):
                    xmed_ul.append(all_preds[i, 0, 1])
                    xmed_ll.append(all_preds[i, 1, 1])
                    xmed_ur.append(all_preds[i, 2, 1])
                    xmed_lr.append(all_preds[i, 3, 1])

                    ymed_ul.append(all_preds[i, 0, 0])
                    ymed_ll.append(all_preds[i, 1, 0])
                    ymed_ur.append(all_preds[i, 2, 0])
                    ymed_lr.append(all_preds[i, 3, 0])

                xs = [
                    int(np.median(xmed_ul)),
                    int(np.median(xmed_ll)),
                    int(np.median(xmed_ur)),
                    int(np.median(xmed_lr)),
                ]
                ys = [
                    int(np.median(ymed_ul)),
                    int(np.median(ymed_ll)),
                    int(np.median(ymed_ur)),
                    int(np.median(ymed_lr)),
                ]
                out_doc = {
                    'corner_coords': {
                        'xs': xs,
                        'ys': ys,
                    }
                }

                video_filename_root, _ = os.path.splitext(video_filename)
                video_yaml_out_filename = video_filename_root + '_corners_v2.yaml'
                print('Writing to:', video_yaml_out_filename)
                with open(video_yaml_out_filename, 'w') as video_yaml_out_file:
                    yaml.safe_dump(out_doc, video_yaml_out_file)

                video_png_out_filename = video_filename_root + '_corners_v2.png'
                for i in range(4):
                    rr, cc = skimage.draw.circle(ys[i], xs[i], 5, mockup.shape)
                    skimage.draw.set_color(mockup, (rr, cc), [255, 0, 0])
                skimage.io.imsave(video_png_out_filename, mockup)


if __name__ == "__main__":
    main()
