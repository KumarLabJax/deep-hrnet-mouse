import argparse
import h5py
import imageio
import numpy as np
import skimage.measure
import time

import torch
import torch.nn.functional as torchfunc
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import _init_paths
from config import cfg
from config import update_config

import models

import matplotlib.pyplot as plt

FRAMES_PER_SECOND = 30
FRAMES_PER_MINUTE = FRAMES_PER_SECOND * 60


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model-file',
        help='the model file to use for inference',
        default=None,
    )

    parser.add_argument(
        '--iou-threshold',
        help='the minimum IOU threshold used for saying two object masks match',
        type=float,
        default=0.95,
    )

    parser.add_argument(
        '--min-obj-size-px',
        help='the minimum object size in pixels',
        type=int,
        default=400,
    )

    parser.add_argument(
        '--minimum-arrangement-duration-secs',
        help='the minimum duration in seconds before an object arrangement is considered valid',
        type=float,
        default=60,
    )

    parser.add_argument(
        '--maximum-merge-duration-secs',
        help='the maximum gap in seconds over which we will try to merge arrangements',
        type=float,
        default=0.5,
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
        'segout',
        help='the segmentation output HDF5 file',
    )

    args = parser.parse_args()
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    if args.model_file:
        cfg.TEST.MODEL_FILE = args.model_file
    cfg.freeze()

    max_merge_frames = round(args.maximum_merge_duration_secs * FRAMES_PER_SECOND)
    min_duration_frames = round(args.minimum_arrangement_duration_secs * FRAMES_PER_SECOND)

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


    def gen_segs():

        with torch.no_grad(), imageio.get_reader(args.video) as reader:
            batch = []
            cuda_segs = None

            def perform_inference():
                nonlocal cuda_segs

                if batch:
                    batch_tensor = torch.stack([xform(img) for img in batch]).cuda()
                    batch.clear()

                    inf_out = model(batch_tensor)
                    in_out_ratio = batch_tensor.size(-1) // inf_out.size(-1)
                    if in_out_ratio == 4:
                        inf_out = torchfunc.upsample(inf_out, scale_factor=4, mode='bicubic', align_corners=False)

                    cuda_segs = inf_out >= 0.0
                    cuda_segs = cuda_segs.squeeze(1)

            for image in reader:

                batch.append(image)
                if len(batch) == cfg.TEST.BATCH_SIZE_PER_GPU:
                    if cuda_segs is not None:
                        cpu_segs = cuda_segs.cpu()
                        cuda_segs = None
                        perform_inference()

                        for i in range(cpu_segs.size(0)):
                            yield cpu_segs[i, ...].numpy()
                    else:
                        perform_inference()

            if cuda_segs is not None:
                cpu_segs = cuda_segs.cpu()
                cuda_segs = None

                for i in range(cpu_segs.size(0)):
                    yield cpu_segs[i, ...].numpy()

            perform_inference()
            if cuda_segs is not None:
                cpu_segs = cuda_segs.cpu()
                cuda_segs = None

                for i in range(cpu_segs.size(0)):
                    yield cpu_segs[i, ...].numpy()

    def gen_accum_masks():
        start_frame = 0
        accum_mask = None
        for frame_index, seg_mask in enumerate(gen_segs()):
            seg_mask = seg_mask.astype(np.bool)

            if accum_mask is None:
                start_frame = frame_index
                accum_mask = np.array(seg_mask, dtype=np.uint32)
            else:
                avg_mask = accum_mask / (frame_index - start_frame) >= 0.5
                sum_avg_mask = avg_mask.sum()
                if sum_avg_mask < args.min_obj_size_px:
                    start_frame = frame_index
                    accum_mask = np.array(seg_mask, dtype=np.uint32)
                else:
                    # perform IOU for the running average object mask vs current seg_mask
                    intersection = np.sum(avg_mask & seg_mask)
                    union = np.sum(avg_mask | seg_mask)
                    curr_iou = intersection / union

                    if curr_iou < args.iou_threshold:
                        # we've passed the threshold of a new object configuration
                        if 1 + frame_index - start_frame >= min_duration_frames:
                            yield accum_mask, start_frame, frame_index

                        start_frame = frame_index
                        accum_mask = np.array(seg_mask, dtype=np.uint32)
                    else:
                        accum_mask += seg_mask

            if frame_index != 0 and frame_index % FRAMES_PER_MINUTE == 0:
                curr_time = time.time()
                cum_time_elapsed = curr_time - start_time
                print('processed {:.1f} min of video in {:.1f} min'.format(
                    frame_index / FRAMES_PER_MINUTE,
                    cum_time_elapsed / 60,
                ))

        if accum_mask is not None:
            frame_index += 1

            avg_mask = accum_mask / (frame_index - start_frame) >= 0.5
            sum_avg_mask = avg_mask.sum()
            if (sum_avg_mask >= args.min_obj_size_px
                    and 1 + frame_index - start_frame >= min_duration_frames):
                yield accum_mask, start_frame, frame_index

    def merge_accum_masks(accum_masks):
        accum_mask = None
        avg_mask = None
        start_frame = 0
        end_frame = 0

        for next_accum_mask, next_start_frame, next_end_frame in accum_masks:
            next_avg_mask = next_accum_mask / (next_end_frame - next_start_frame) >= 0.5

            merge_happened = False
            if accum_mask is not None:
                frame_gap = next_start_frame - end_frame
                if frame_gap <= max_merge_frames:
                    # perform IOU for the masks
                    intersection = np.sum(avg_mask & next_avg_mask)
                    union = np.sum(avg_mask | next_avg_mask)
                    curr_iou = intersection / union
                    if curr_iou >= args.iou_threshold:
                        # we can perform the merge
                        accum_mask += next_accum_mask
                        end_frame = next_end_frame
                        merge_happened = True

                if not merge_happened:
                    yield accum_mask, start_frame, end_frame

            if not merge_happened:
                accum_mask = next_accum_mask
                avg_mask = next_avg_mask
                start_frame = next_start_frame
                end_frame = next_end_frame

        if accum_mask is not None:
            yield accum_mask, start_frame, end_frame

    with h5py.File(args.segout, 'w') as segout_h5:
        seg_group_index = 0
        accum_masks = gen_accum_masks()
        if max_merge_frames > 0:
            accum_masks = merge_accum_masks(accum_masks)

        for accum_mask, start_frame, end_frame in accum_masks:
            print('start frame:', start_frame, 'end frame:', end_frame)
            avg_mask = accum_mask / (end_frame - start_frame) >= 0.5

            labels, label_count = skimage.measure.label(avg_mask, return_num=True)
            obj_mask_gen = (labels == i for i in range(1, label_count + 1))
            obj_masks = [mask.astype(np.uint8) for mask in obj_mask_gen if mask.sum() >= args.min_obj_size_px]

            if obj_masks:
                mask_ds_path = 'objectsegs/arrangement_' + str(seg_group_index)
                segout_h5[mask_ds_path] = np.stack(obj_masks)
                segout_h5[mask_ds_path].attrs['start_frame'] = start_frame
                segout_h5[mask_ds_path].attrs['end_frame'] = end_frame
                seg_group_index += 1


if __name__ == "__main__":
    main()
