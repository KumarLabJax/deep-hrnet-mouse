import argparse
import imageio
import itertools
import numpy as np
import os
import yaml

import torch
import torch.backends.cudnn as cudnn

import _init_paths
from config import cfg
from config import update_config

from inferfecalbolicount import infer_fecal_boli_counts

import models

# Examples:
#
#   python -u tools/inferfecalbolicountbatch.py \
#       --root-dir '/media/sheppk/TOSHIBA EXT/cached-data/BTBR_3M_stranger_4day' \
#       --batch-file '/media/sheppk/TOSHIBA EXT/cached-data/BTBR_3M_stranger_4day/BTBR_3M_stranger_4day-batch-temp.txt' \
#       output-fecal-boli/fecalboli/pose_hrnet/fecalboli_2020-06-19_02/best_state.pth \
#       experiments/fecalboli/fecalboli_2020-06-19_02.yaml
#
#   python -u tools/inferfecalbolicountbatch.py \
#       --root-dir '/home/sheppk/smb/labshare' \
#       --batch-file 'data/fecal-boli/Tom-CBAX2B-OFA_batch.txt' \
#       output-fecal-boli/fecalboli/pose_hrnet/fecalboli_2020-06-19_02/best_state.pth \
#       experiments/fecalboli/fecalboli_2020-06-19_02.yaml

# Strain Survey:
#   python -u tools/inferfecalbolicountbatch.py \
#       --allow-missing-video \
#       --root-dir '/home/sheppk/smb/labshare' \
#       --batch-file 'data/fecal-boli/strain-survey-batch-2019-05-29.txt' \
#       output-fecal-boli/fecalboli/pose_hrnet/fecalboli_2020-06-19_02/best_state.pth \
#       experiments/fecalboli/fecalboli_2020-06-19_02.yaml

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model_file',
        help='the model file to use for inference',
    )
    parser.add_argument(
        'cfg',
        help='the configuration for the model to use for inference',
    )
    parser.add_argument(
        '--batch-file',
        help='path to the file that is a new-line separated'
             ' list of all videos to process',
        required=True,
    )
    parser.add_argument(
        '--root-dir',
        help='the root directory. All paths given in the batch files are relative to this root',
        required=True,
    )
    parser.add_argument(
        '--min-heatmap-val',
        type=float,
        default=0.75,
    )
    parser.add_argument(
        '--sample-interval-min',
        type=int,
        default=1,
        help='what sampling interval should we use for frames',
    )
    parser.add_argument(
        '--allow-missing-video',
        help='allow missing videos with warning',
        action='store_true',
    )

    args = parser.parse_args()
    sample_intervals_frames = args.sample_interval_min * 60 * 30

    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.TEST.MODEL_FILE = args.model_file
    cfg.freeze()

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

    with open(args.batch_file) as batch_file:
        for line in batch_file:
            vid_filename = line.strip()
            if vid_filename:
                vid_path = os.path.join(args.root_dir, vid_filename)
                vid_path_root, _ = os.path.splitext(vid_path)
                vid_fb_count_path = vid_path_root + '_fecal_boli_counts.yaml'
                print(vid_fb_count_path)

                if args.allow_missing_video:
                    if not os.path.exists(vid_path):
                        print('WARNING: ' + vid_path + ' does not exist')
                        continue
                else:
                    assert os.path.exists(vid_path), vid_path + ' does not exist'

                with imageio.get_reader(vid_path) as frame_reader:

                    frame_reader = itertools.islice(frame_reader, 0, None, sample_intervals_frames)
                    fecal_boli_counts = infer_fecal_boli_counts(model, frame_reader, args.min_heatmap_val)
                    fecal_boli_counts = list(fecal_boli_counts)
                    out_doc = {
                        'sample_interval_min': args.sample_interval_min,
                        'fecal_boli_counts': fecal_boli_counts,
                    }
                    with open(vid_fb_count_path, 'w') as video_yaml_out_file:
                        yaml.safe_dump(out_doc, video_yaml_out_file)


if __name__ == "__main__":
    main()
