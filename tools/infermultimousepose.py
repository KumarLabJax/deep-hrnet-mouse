import argparse
import h5py
import imageio
import numpy as np
import skimage.transform
import time

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
#   python -u tools/infermultimousepose.py \
#       --max-instance-count 3 \
#       ./output-multi-mouse/multimousepose/pose_hrnet/multimouse_2019-11-19_1/best_state.pth \
#       ./experiments/multimouse/multimouse_2019-11-19_1.yaml \
#       one-min-clip-800x800.avi \
#       one-min-clip-800x800_2019-11-19_2.h5
#
#   python -u tools/infermultimousepose.py \
#       --max-instance-count 4 \
#       ./output-multi-mouse/multimousepose/pose_hrnet/multimouse_2019-11-19_1/best_state.pth \
#       ./experiments/multimouse/multimouse_2019-11-19_1.yaml \
#       one-min-clip-5.avi \
#       one-min-clip-5-2.h5
#
#   python -u tools/infermultimousepose.py \
#       --max-instance-count 4 \
#       --max-embed-sep-within-instances 0.3 \
#       --min-embed-sep-between-instances 0.2 \
#       --min-pose-heatmap-val 1.5 \
#       ./output-multi-mouse/multimousepose/pose_hrnet/multimouse_2020-02-03_06/best_state.pth \
#       ./experiments/multimouse/multimouse_2020-02-03_06.yaml \
#       one-min-clip-4.avi \
#       one-min-clip-4-2020-02-03_06-WEIGHTED_EMBED-HIGHER_PROB.h5

def infer_pose_instances(
        model, frames,
        use_neighboring_frames,
        min_embed_sep, max_embed_sep, max_inst_dist,
        min_joint_count, max_instance_count, max_pose_dist_px,
        min_pose_heatmap_val):

    def infer_pose_instances_no_track_id():

        xform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.45, 0.45, 0.45],
                std=[0.225, 0.225, 0.225],
            ),
        ])

        with torch.no_grad():

            start_time = time.time()

            # Build up a list of lists containing PoseInstance objects. The elements
            # in pose_instances correspond to video frames and the indices of the
            # nested lists correspond to instances detected within the respective frame.
            # pose_instances = []

            batch = []
            cuda_pose_heatmap = None
            cuda_pose_localmax = None
            cuda_pose_embed_map = None

            def sync_cuda_preds():
                nonlocal cuda_pose_heatmap
                nonlocal cuda_pose_localmax
                nonlocal cuda_pose_embed_map

                batch_pose_instances = []

                if cuda_pose_heatmap is not None:
                    # calculate pose instances and add them to pose_instances list
                    curr_batch_size = cuda_pose_heatmap.size(0)
                    for batch_frame_index in range(curr_batch_size):

                        # pylint: disable=unsubscriptable-object
                        frame_pose_instances = aeutil.calc_pose_instances(
                            cuda_pose_heatmap[batch_frame_index, ...],
                            cuda_pose_localmax[batch_frame_index, ...],
                            cuda_pose_embed_map[batch_frame_index, ...],
                            min_embed_sep,
                            max_embed_sep,
                            max_inst_dist)

                        # remove poses that have too few joints
                        if min_joint_count is not None:
                            frame_pose_instances = [
                                pi for pi in frame_pose_instances
                                if len(pi.keypoints) >= min_joint_count
                            ]

                        # if we have too many poses remove in order of lowest confidence
                        if (max_instance_count is not None
                                and len(frame_pose_instances) > max_instance_count):
                            frame_pose_instances.sort(key=lambda pi: pi.mean_inst_conf)
                            del frame_pose_instances[max_instance_count:]

                        batch_pose_instances.append(frame_pose_instances)

                    cuda_pose_heatmap = None
                    cuda_pose_localmax = None
                    cuda_pose_embed_map = None

                return batch_pose_instances

            def perform_inference():
                nonlocal cuda_pose_heatmap
                nonlocal cuda_pose_localmax
                nonlocal cuda_pose_embed_map

                prev_batch_pose_instances = None

                if batch:
                    batch_tensor = torch.stack(batch[:cfg.TEST.BATCH_SIZE_PER_GPU])
                    del batch[:cfg.TEST.BATCH_SIZE_PER_GPU]
                    batch_tensor = batch_tensor.cuda(non_blocking=True)

                    prev_batch_pose_instances = sync_cuda_preds()

                    model_out = model(batch_tensor)

                    joint_count = model_out.size(1) // 2
                    cuda_pose_heatmap = model_out[:, :joint_count, ...]
                    cuda_pose_localmax = aeutil.localmax2D(cuda_pose_heatmap, min_pose_heatmap_val, 3)
                    cuda_pose_embed_map = model_out[:, joint_count:, ...]
                else:
                    prev_batch_pose_instances = sync_cuda_preds()

                return prev_batch_pose_instances

            for frame_index, image in enumerate(frames):

                if frame_index != 0 and frame_index % (FRAMES_PER_MINUTE // 4) == 0:
                    curr_time = time.time()
                    cum_time_elapsed = curr_time - start_time
                    print('processed {:.2f} min of video in {:.2f} min'.format(
                        frame_index / FRAMES_PER_MINUTE,
                        cum_time_elapsed / 60,
                    ))

                image = xform(image)

                prev_batch_pose_instances = []
                if use_neighboring_frames:
                    if len(batch) >= 1:
                        image[0, ...] = batch[-1][1, ...]
                        batch[-1][2, ...] = image[1, ...]
                    if len(batch) == cfg.TEST.BATCH_SIZE_PER_GPU + 1:
                        prev_batch_pose_instances = perform_inference()
                else:
                    batch.append(image)
                    if len(batch) == cfg.TEST.BATCH_SIZE_PER_GPU:
                        prev_batch_pose_instances = perform_inference()

                for frame_pose_instances in prev_batch_pose_instances:
                    yield frame_pose_instances

            # In this while loop we drain any remaining batchs. It should iterate
            # at most two times.
            prev_batch_pose_instances = perform_inference()
            while prev_batch_pose_instances:
                for frame_pose_instances in prev_batch_pose_instances:
                    yield frame_pose_instances
                prev_batch_pose_instances = perform_inference()

    return apply_track_id_to_poses(max_pose_dist_px, infer_pose_instances_no_track_id())


def apply_track_id_to_poses(max_pose_dist_px, pose_instances):
    # we now have a collection of pose instances for every frame. We can try
    # to join them together into tracks based on pose distance.
    track_id_counter = 0
    prev_pose_instances = []
    for curr_pose_instances in pose_instances:
        pose_combos = []
        for prev_pose_i, prev_pose in enumerate(prev_pose_instances):
            for curr_pose_i, curr_pose in enumerate(curr_pose_instances):
                curr_dist = aeutil.pose_distance(curr_pose, prev_pose)
                if curr_dist <= max_pose_dist_px:
                    pose_combos.append((prev_pose_i, curr_pose_i, curr_dist))

        # sort pose combinations by distance
        pose_combos.sort(key=lambda pcombo: pcombo[2])

        unmatched_prev_poses = set(range(len(prev_pose_instances)))
        unmatched_curr_poses = set(range(len(curr_pose_instances)))
        for prev_pose_i, curr_pose_i, curr_dist in pose_combos:
            if prev_pose_i in unmatched_prev_poses and curr_pose_i in unmatched_curr_poses:
                prev_pose = prev_pose_instances[prev_pose_i]
                curr_pose = curr_pose_instances[curr_pose_i]
                curr_pose.instance_track_id = prev_pose.instance_track_id

                unmatched_prev_poses.remove(prev_pose_i)
                unmatched_curr_poses.remove(curr_pose_i)

        for unmatched_pose_i in unmatched_curr_poses:
            curr_pose = curr_pose_instances[unmatched_pose_i]
            curr_pose.instance_track_id = track_id_counter
            track_id_counter += 1

        prev_pose_instances = curr_pose_instances

        yield curr_pose_instances


# def resize_frames(frames, height, width):
#     for frame in frames:
#         print('BEFORE dtype, shape, min, max:', frame.dtype, frame.shape, frame.min(), frame.max())
#         frame = skimage.transform.resize(frame, (height, width))
#         frame = np.round(frame * 255).astype(np.uint8)
#
#         print('AFTER  dtype, shape, min, max:', frame.dtype, frame.shape, frame.min(), frame.max())
#
#         yield frame


def find_same_track_pose(pose, pose_list):
    for curr_pose in pose_list:
        if curr_pose.instance_track_id == pose.instance_track_id:
            return curr_pose

    return None


def smooth_poses(pose_instances):
    frame_count = len(pose_instances)
    for frame_index, curr_frame_pose_instances in enumerate(pose_instances):

        prev_frame_pose_instances = []
        if frame_index > 0:
            prev_frame_pose_instances = pose_instances[frame_index - 1]

        next_frame_pose_instances = []
        if frame_index < frame_count - 1:
            next_frame_pose_instances = pose_instances[frame_index + 1]

        for curr_pose_track_instance in curr_frame_pose_instances:
            prev_pose_track_instance = find_same_track_pose(
                curr_pose_track_instance,
                prev_frame_pose_instances)
            next_pose_track_instance = find_same_track_pose(
                curr_pose_track_instance,
                next_frame_pose_instances)

            # we only try to smooth if we have both prev and next pose
            if prev_pose_track_instance is not None and next_pose_track_instance is not None:
                curr_joint_indexes = curr_pose_track_instance.keypoints.keys()
                prev_joint_indexes = prev_pose_track_instance.keypoints.keys()
                next_joint_indexes = next_pose_track_instance.keypoints.keys()

                # we only try to smooth if we have curr, prev and next keypoints
                joints_to_smooth = curr_joint_indexes & prev_joint_indexes & next_joint_indexes

                for joint_index in joints_to_smooth:
                    prev_keypoint = prev_pose_track_instance.keypoints[joint_index]
                    next_keypoint = next_pose_track_instance.keypoints[joint_index]

                    curr_keypoint = curr_pose_track_instance.keypoints[joint_index]
                    curr_keypoint['x_pos'] = round(
                        (curr_keypoint['x_pos'] + prev_keypoint['x_pos'] + next_keypoint['x_pos']) / 3.0)
                    curr_keypoint['y_pos'] = round(
                        (curr_keypoint['y_pos'] + prev_keypoint['y_pos'] + next_keypoint['y_pos']) / 3.0)


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
        'poseout',
        help='the pose estimation output HDF5 file',
    )
    # TODO we should change this to cm units rather than pixels
    parser.add_argument(
        '--max-inst-dist-px',
        help='maximum keypoint separation distance in pixels. For a keypoint to '
             'be added to an instance there must be at least one point in the '
             'instance which is within this number of pixels.',
        type=int,
        default=150,
    )
    parser.add_argument(
        '--max-embed-sep-within-instances',
        help='maximum embedding separation allowed for a joint to be added to an existing '
             'instance within the max distance separation',
        type=float,
        default=0.2,
    )
    parser.add_argument(
        '--min-embed-sep-between-instances',
        help='if two joints of the the same type (eg. both right ear) are within the max '
             'distance separation and their embedding separation doesn\'t meet or '
             'exceed this threshold only the point with the highest heatmap value is kept.',
        type=float,
        default=0.1,
    )
    parser.add_argument(
        '--min-pose-heatmap-val',
        type=float,
        default=0.4,
    )
    parser.add_argument(
        '--max-pose-dist-px',
        type=float,
        default=40,
    )
    parser.add_argument(
        '--min-joint-count',
        help='if a pose instance has fewer than this number of points it is discarded',
        type=int,
        default=6,
    )
    parser.add_argument(
        '--max-instance-count',
        help='a frame should not contain more than this number of poses. If it does, extra poses '
             'will be discarded in order of least confidence until we meet this threshold.',
        type=int,
    )
    parser.add_argument(
        '--pose-smoothing',
        help='apply a smoothing to the pose by averaging position over three frames',
        action='store_true',
    )

    args = parser.parse_args()

    # shorten some args
    max_embed_sep = args.max_embed_sep_within_instances
    min_embed_sep = args.min_embed_sep_between_instances
    max_inst_dist = args.max_inst_dist_px

    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.TEST.MODEL_FILE = args.model_file
    cfg.freeze()

    # start_time = time.time()

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

    model_extra = cfg.MODEL.EXTRA
    use_neighboring_frames = False
    if 'USE_NEIGHBORING_FRAMES' in model_extra:
        use_neighboring_frames = model_extra['USE_NEIGHBORING_FRAMES']

    with imageio.get_reader(args.video) as frame_reader:

        # if args.resize_frames:
        #     resize_height, resize_width = args.resize_frames
        #     frame_reader = resize_frames(frame_reader, resize_height, resize_width)

        pose_instances = list(infer_pose_instances(
                model, frame_reader,
                use_neighboring_frames,
                min_embed_sep, max_embed_sep, max_inst_dist,
                args.min_joint_count, args.max_instance_count, args.max_pose_dist_px,
                args.min_pose_heatmap_val))
        frame_count = len(pose_instances)

        # remove points that jump too much since they are likely to be errors. A point is only
        # considered a "jump" if it's distance is too great for the previous and next frame
        # position of the corresponding point.
        for frame_index in range(frame_count):
            curr_frame_pose_instances = pose_instances[frame_index]

            prev_frame_pose_instances = []
            if frame_index > 0:
                prev_frame_pose_instances = pose_instances[frame_index - 1]

            next_frame_pose_instances = []
            if frame_index < frame_count - 1:
                next_frame_pose_instances = pose_instances[frame_index + 1]

            for curr_pose_track_instance in curr_frame_pose_instances:
                prev_pose_track_instance = find_same_track_pose(
                    curr_pose_track_instance,
                    prev_frame_pose_instances)
                next_pose_track_instance = find_same_track_pose(
                    curr_pose_track_instance,
                    next_frame_pose_instances)

                if prev_pose_track_instance is not None or next_pose_track_instance is not None:
                    for keypoint in list(curr_pose_track_instance.keypoints.values()):
                        prev_next_dists = []

                        try:
                            prev_keypoint = prev_pose_track_instance.keypoints[keypoint['joint_index']]
                            prev_next_dists.append(aeutil.xy_dist(prev_keypoint, keypoint))
                        except:
                            pass

                        try:
                            next_keypoint = next_pose_track_instance.keypoints[keypoint['joint_index']]
                            prev_next_dists.append(aeutil.xy_dist(next_keypoint, keypoint))
                        except:
                            pass

                        # here is where we remove the point if it's too far from it's neighbors
                        if prev_next_dists:
                            if all(dist > args.max_pose_dist_px for dist in prev_next_dists):
                                del curr_pose_track_instance.keypoints[keypoint['joint_index']]

        # get rid of poses that don't meet our point count threshold
        for frame_index in range(frame_count):
            curr_frame_pose_instances = pose_instances[frame_index]
            pose_instances[frame_index] = [
                p for p in curr_frame_pose_instances
                if len(p.keypoints) >= args.min_joint_count
            ]

        # get rid of "isolated" poses (where there is no previous or next pose
        # with the same track ID)
        for frame_index in range(frame_count):
            curr_frame_pose_instances = pose_instances[frame_index]

            prev_frame_pose_instances = []
            if frame_index > 0:
                prev_frame_pose_instances = pose_instances[frame_index - 1]

            next_frame_pose_instances = []
            if frame_index < frame_count - 1:
                next_frame_pose_instances = pose_instances[frame_index + 1]

            pose_instances[frame_index] = [
                p for p in curr_frame_pose_instances

                # if not isolated
                if
                    find_same_track_pose(p, prev_frame_pose_instances) is not None or
                    find_same_track_pose(p, next_frame_pose_instances) is not None
            ]

        # now that we've made all of these changes we should update the track ids
        pose_instances = list(apply_track_id_to_poses(args.max_pose_dist_px, pose_instances))

        if args.pose_smoothing:
            smooth_poses(pose_instances)

        max_instance_count = 0
        for curr_pose_instances in pose_instances:

            if len(curr_pose_instances) > max_instance_count:
                max_instance_count = len(curr_pose_instances)

            # print(
            #     'pose_count:', len(curr_pose_instances),
            #     'track_ids:', ' '.join([
            #         str(p.instance_track_id)
            #         for p
            #         in sorted(curr_pose_instances, key=lambda pose: pose.instance_track_id)]))

        # save data to an HDF5 file
        points = np.zeros(
            (frame_count, max_instance_count, 12, 2),
            dtype=np.uint16)
        confidence = np.zeros(
            (frame_count, max_instance_count, 12),
            dtype=np.float32)
        instance_count = np.zeros(
            frame_count,
            dtype=np.uint8)
        embed = np.zeros(
            (frame_count, max_instance_count, 12),
            dtype=np.float32)
        instance_track_id = np.zeros(
            (frame_count, max_instance_count),
            dtype=np.uint32)

        for frame_index, frame_pose_instances in enumerate(pose_instances):
            instance_count[frame_index] = len(frame_pose_instances)
            for pose_index, pose_instance in enumerate(frame_pose_instances):
                instance_track_id[frame_index, pose_index] = pose_instance.instance_track_id
                for keypoint in pose_instance.keypoints.values():
                    points[frame_index, pose_index, keypoint['joint_index'], 0] = keypoint['y_pos']
                    points[frame_index, pose_index, keypoint['joint_index'], 1] = keypoint['x_pos']
                    confidence[frame_index, pose_index, keypoint['joint_index']] = keypoint['conf']
                    embed[frame_index, pose_index, keypoint['joint_index']] = keypoint['embed']

        with h5py.File(args.poseout, 'w') as h5file:
            h5file['poseest/points'] = points
            h5file['poseest/confidence'] = confidence
            h5file['poseest/instance_count'] = instance_count
            h5file['poseest/instance_embedding'] = embed
            h5file['poseest/instance_track_id'] = instance_track_id

            h5file['poseest'].attrs['version'] = np.array([3, 0], dtype=np.uint16)


if __name__ == "__main__":
    main()
