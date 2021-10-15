import matplotlib
matplotlib.use("Agg")

import argparse
import imageio
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import skimage.transform

import torch
import torch.nn.functional as torchfunc
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import _init_paths
import utils.assocembedutil as aeutil
from config import cfg
from config import update_config

import models


FRAMES_PER_MINUTE = 30 * 60

# Examples:
#
#   python -u tools/inferfecalbolicount.py \
#       --min-heatmap-val 1.5 \
#       output-fecal-boli/fecalboli/pose_hrnet/fecalboli_2020-05-0-08/best_state.pth \
#       experiments/fecalboli/fecalboli_2020-05-0-08.yaml \
#       one-min-clip-4.avi

#   python -u tools/inferfecalbolicount.py \
#       --min-heatmap-val 0.75 \
#       --image-out-dir temp/fbinf \
#       output-fecal-boli/fecalboli/pose_hrnet/fecalboli_2020-06-19_02/best_state.pth \
#       experiments/fecalboli/fecalboli_2020-06-19_02.yaml \
#       one-min-clip-4.avi

#     for i in `ls poseintervals-temp/*.avi`
#     do
#         echo "PROCESSING ${i}"
#         python -u tools/inferfecalbolicount.py \
#             --min-heatmap-val 0.75 \
#             --image-out-dir "poseintervals-temp" \
#             output-fecal-boli/fecalboli/pose_hrnet/fecalboli_2020-06-19_02/best_state.pth \
#             experiments/fecalboli/fecalboli_2020-06-19_02.yaml \
#             "${i}"
#     done

def infer_fecal_boli_xy(
        model,
        frames,
        min_heatmap_val,
        image_out_dir=None,
        image_name_prefix='fb-inf-'):

    xform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.45, 0.45, 0.45],
            std=[0.225, 0.225, 0.225],
        ),
    ])

    with torch.no_grad():

        batch = []
        batch_images = []
        cuda_heatmap = None
        cuda_localmax = None
        image_save_counter = 0

        def sync_cuda_preds():
            nonlocal cuda_heatmap
            nonlocal cuda_localmax
            nonlocal image_save_counter
            
            # print('=== sync_cuda_preds ===')
            batch_fecal_boli_xy = []

            if cuda_heatmap is not None:
                # calculate fecal boli XY and add them to batch_fecal_boli_xy list
                curr_batch_size = cuda_heatmap.size(0)
                for batch_frame_index in range(curr_batch_size):

                    frame_cuda_localmax = cuda_localmax[batch_frame_index, 0, ...]
                    frame_fecal_boli_xy = frame_cuda_localmax.nonzero().cpu()
                    frame_fecal_boli_xy[...] = frame_fecal_boli_xy[..., [1, 0]].clone()

                    batch_fecal_boli_xy.append(frame_fecal_boli_xy)

                    if image_out_dir is not None:
                        os.makedirs(image_out_dir, exist_ok=True)

                        data_numpy = batch_images.pop(0)

                        curr_heatmap = cuda_heatmap[batch_frame_index, 0, ...].cpu().numpy()
                        plt.figure(figsize=(12, 12))
                        plt.imshow(curr_heatmap, cmap=plt.get_cmap('YlOrRd'))
                        plt.savefig(os.path.join(image_out_dir, image_name_prefix + 'heat-' + str(image_save_counter) + '.png'))

                        curr_heatmap = np.ma.masked_where(curr_heatmap <= min_heatmap_val, curr_heatmap)
                        fig = plt.figure(figsize=(24, 12))
                        ax = fig.gca()
                        plt.imshow(np.concatenate((data_numpy, data_numpy), axis=1), cmap='gray', vmin=0, vmax=255)
                        plt.imshow(curr_heatmap, cmap=plt.get_cmap('YlOrRd'), alpha=0.4)

                        image_width = data_numpy.shape[1]
                        for curr_xy in frame_fecal_boli_xy + torch.tensor([image_width, 0]):
                            ax.add_artist(plt.Circle(curr_xy, 10, color='r', fill=False))
                        plt.imshow(np.concatenate((data_numpy, data_numpy), axis=1), cmap='gray', vmin=0, vmax=255, alpha=0.0)
                        plt.axis('off')
                        plt.tight_layout()
                        plt.savefig(os.path.join(image_out_dir, image_name_prefix + str(image_save_counter) + '.png'))
                        image_save_counter += 1
                        del batch_images[:cfg.TEST.BATCH_SIZE_PER_GPU]

                cuda_heatmap = None
                cuda_localmax = None

            return batch_fecal_boli_xy

        def perform_inference():
            nonlocal cuda_heatmap
            nonlocal cuda_localmax

            prev_batch_fecal_boli_xy = None

            if batch:
                batch_tensor = torch.stack(batch[:cfg.TEST.BATCH_SIZE_PER_GPU])
                del batch[:cfg.TEST.BATCH_SIZE_PER_GPU]
                batch_tensor = batch_tensor.cuda(non_blocking=True)

                prev_batch_fecal_boli_xy = sync_cuda_preds()

                model_out = model(batch_tensor)

                cuda_heatmap = model_out
                cuda_localmax = aeutil.localmax2D(cuda_heatmap, min_heatmap_val, 5)
            else:
                prev_batch_fecal_boli_xy = sync_cuda_preds()

            return prev_batch_fecal_boli_xy

        for frame_index, image in enumerate(frames):

            if image_out_dir is not None:
                batch_images.append(image)

            image = xform(image)

            prev_batch_fecal_boli_xy = []
            batch.append(image)
            if len(batch) == cfg.TEST.BATCH_SIZE_PER_GPU:
                prev_batch_fecal_boli_xy = perform_inference()

            for frame_fecal_boli_xy in prev_batch_fecal_boli_xy:
                yield frame_fecal_boli_xy

        # Drain any remaining batchs. It should require at most two calls.
        for _ in range(2):
            prev_batch_fecal_boli_xy = perform_inference()
            for frame_fecal_boli_xy in prev_batch_fecal_boli_xy:
                yield frame_fecal_boli_xy


def infer_fecal_boli_counts(model, frames, min_heatmap_val, image_out_dir=None, image_name_prefix='fb-inf-'):
    fecal_boli_xys = infer_fecal_boli_xy(model, frames, min_heatmap_val, image_out_dir, image_name_prefix)
    for fecal_boli_xy in fecal_boli_xys:
        yield fecal_boli_xy.size(0)


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
        'video',
        help='the input video',
    )
    parser.add_argument(
        '--image-out-dir',
        type=str,
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

    with imageio.get_reader(args.video) as frame_reader:

        out_prefex, _ = os.path.splitext(os.path.basename(args.video))
        out_prefex += '_fecal_boli_'

        frame_reader = itertools.islice(frame_reader, 0, None, sample_intervals_frames)
        fecal_boli_counts = infer_fecal_boli_counts(
            model,
            frame_reader,
            args.min_heatmap_val,
            args.image_out_dir,
            out_prefex)
        for counts in fecal_boli_counts:
            print('counts:', counts)


if __name__ == "__main__":
    main()
