import argparse
import cv2
import imageio
import multiprocessing as mp
import numpy as np
import os
import h5py


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


CONNECTED_SEGMENTS = [
        [LEFT_FRONT_PAW_INDEX, CENTER_SPINE_INDEX, RIGHT_FRONT_PAW_INDEX],
        [LEFT_REAR_PAW_INDEX, BASE_TAIL_INDEX, RIGHT_REAR_PAW_INDEX],
        [
            NOSE_INDEX, BASE_NECK_INDEX, CENTER_SPINE_INDEX,
            BASE_TAIL_INDEX, MID_TAIL_INDEX, TIP_TAIL_INDEX,
        ],
]

# from: http://colorbrewer2.org/?type=qualitative&scheme=Set3&n=8
# COLOR_PALETTE = [
#     (141,211,199),
#     (255,255,179),
#     (190,186,218),
#     (251,128,114),
#     (128,177,211),
#     (253,180,98),
#     (179,222,105),
#     (252,205,229),
# ]

COLOR_PALETTE = [
    (166,206,227),
    (31,120,180),
    (178,223,138),
    (51,160,44),
    (251,154,153),
    (227,26,28),
    (253,191,111),
    (255,127,0),
    (202,178,214),
    (106,61,154),
    (255,255,153)]

def render_pose_overlay(image, frame_points, exclude_points, color=(255, 255, 255)):

    # we need to fragment lines if exclude_points breaks up
    # (or removes completely) line segments
    def gen_line_fragments():
        curr_fragment = []
        for curr_pt_indexes in CONNECTED_SEGMENTS:
            for curr_pt_index in curr_pt_indexes:
                if curr_pt_index in exclude_points:
                    if len(curr_fragment) >= 2:
                        yield curr_fragment
                    curr_fragment = []
                else:
                    curr_fragment.append(curr_pt_index)

            if len(curr_fragment) >= 2:
                yield curr_fragment
            curr_fragment = []
    line_pt_indexes = list(gen_line_fragments())

    for curr_line_indexes in line_pt_indexes:
        line_pts = np.array(
            [(pt_x, pt_y) for pt_y, pt_x in frame_points[curr_line_indexes]],
            np.int32)
        cv2.polylines(image, [line_pts], False, (0, 0, 0), 2, cv2.LINE_AA)

    for point_index in range(12):
        if point_index in exclude_points:
            continue

        point_y, point_x = frame_points[point_index, :]

        cv2.circle(image, (point_x, point_y), 3, (0, 0, 0), -1, cv2.LINE_AA)

    for curr_line_indexes in line_pt_indexes:
        line_pts = np.array(
            [(pt_x, pt_y) for pt_y, pt_x in frame_points[curr_line_indexes]],
            np.int32)
        cv2.polylines(image, [line_pts], False, color, 1, cv2.LINE_AA)

    for point_index in range(12):
        if point_index in exclude_points:
            continue

        point_y, point_x = frame_points[point_index, :]

        cv2.circle(image, (point_x, point_y), 2, color, -1, cv2.LINE_AA)


def render_pose_v3_overlay(
        image,
        frame_points,
        frame_confidence,
        frame_track_ids,
        exclude_points):

    instance_count = frame_points.shape[0]

    id_color_dict = dict()
    avail_color_idxs = set(range(len(COLOR_PALETTE)))
    sorted_ids = sorted(frame_track_ids)

    if len(frame_track_ids) <= len(COLOR_PALETTE):
        for curr_id in sorted_ids:
            curr_color_idx = curr_id % len(COLOR_PALETTE)
            offset = 0
            while curr_color_idx not in avail_color_idxs:
                offset += 1
                curr_color_idx = (curr_id + offset) % len(COLOR_PALETTE)

            id_color_dict[curr_id] = COLOR_PALETTE[curr_color_idx]
            avail_color_idxs.remove(curr_color_idx)
    else:
        id_color_dict = {i: (255, 255, 255) for i in sorted_ids}

    for instance_index in range(instance_count):

        # for this instance we add zero confidence points to the
        # set of excluded point indexes
        inst_confidence = frame_confidence[instance_index, :]
        zero_conf_indexes = set((inst_confidence == 0).nonzero()[0])
        inst_exclude_points = exclude_points | zero_conf_indexes

        render_pose_overlay(
            image,
            frame_points[instance_index, ...],
            inst_exclude_points,
            id_color_dict[frame_track_ids[instance_index]])


def process_video(in_video_path, pose_h5_path, out_video_path, exclude_points):
    if not os.path.isfile(in_video_path):
        print('ERROR: missing file: ' + in_video_path, flush=True)
        return

    if not os.path.isfile(pose_h5_path):
        print('ERROR: missing file: ' + pose_h5_path, flush=True)
        return

    with imageio.get_reader(in_video_path) as video_reader, \
        h5py.File(pose_h5_path, 'r') as pose_h5, \
        imageio.get_writer(out_video_path, fps=30) as video_writer:

        vid_grp = next(iter(pose_h5.values()))
        major_version = 2
        if 'version' in vid_grp.attrs:
            major_version = vid_grp.attrs['version'][0]

        if major_version == 2:
            all_points = vid_grp['points'][:]
            for frame_index, image in enumerate(video_reader):

                render_pose_overlay(
                    image,
                    all_points[frame_index, ...],
                    exclude_points)

                video_writer.append_data(image)

        elif major_version == 3:
            all_points = vid_grp['points'][:]
            all_confidence = vid_grp['confidence'][:]
            all_instance_count = vid_grp['instance_count'][:]
            all_track_id = vid_grp['instance_track_id'][:]
            for frame_index, image in enumerate(video_reader):

                frame_instance_count = all_instance_count[frame_index]
                if frame_instance_count > 0:
                    render_pose_v3_overlay(
                        image,
                        all_points[frame_index, :frame_instance_count, ...],
                        all_confidence[frame_index, :frame_instance_count, ...],
                        all_track_id[frame_index, :frame_instance_count],
                        exclude_points)

                video_writer.append_data(image)

        else:
            print('ERROR: unknown version for file format:', vid_grp.attrs['version'])

    print('finished generating video:', out_video_path, flush=True)


def process_video_relpath(video_relpath, pose_suffix, in_dir, out_dir, exclude_points):

    pose_suffex_noext, _ = os.path.splitext(pose_suffix)
    if len(pose_suffex_noext) == 0:
        print('ERROR: bad pose suffix: ' + pose_suffix, flush=True)
        return

    # calculate full file paths from the in/out dirs and relative path
    relpath_noext, _ = os.path.splitext(video_relpath)
    in_video_path = os.path.join(in_dir, video_relpath)
    pose_h5_path = os.path.join(in_dir, relpath_noext + pose_suffix)
    out_video_path = os.path.join(out_dir, relpath_noext + pose_suffex_noext + '.avi')

    # we may need to create the output dir
    if out_dir != in_dir:
        full_out_dir = os.path.dirname(out_video_path)
        os.makedirs(full_out_dir, exist_ok=True)

    process_video(in_video_path, pose_h5_path, out_video_path, exclude_points)


# Examples:
#   python -u tools/rendervidoverlay.py \
#       --exclude-forepaws --exclude-ears \
#       dir --in-dir ~/smb/labshare \
#       --pose-suffix '_pose_est_v3.h5' --num-procs 3 \
#       --out-dir ~/smb/labshare/kumarlab-new/Keith/BXD-pose-overlay-2020-08-14 \
#       --batch-file data/BXD-batch-50-subset.txt

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--exclude-forepaws',
        action='store_true',
        dest='exclude_forepaws',
        default=False,
        help='should we exclude the forepaws',
    )
    parser.add_argument(
        '--exclude-ears',
        action='store_true',
        dest='exclude_ears',
        default=False,
        help='should we exclude the ears',
    )

    subparsers = parser.add_subparsers()

    dir_parser = subparsers.add_parser(
        'dir',
        help='dir subcommand help (for processing a directory of videos)')
    dir_parser.set_defaults(subcommand='dir')

    dir_parser.add_argument(
        '--in-dir',
        help='input directory of videos to process',
        required=True,
    )
    dir_parser.add_argument(
        '--out-dir',
        help='out directory to save videos to (defaults to the same as --in-dir)',
        required=False,
    )
    dir_parser.add_argument(
        '--pose-suffix',
        help='the suffix used for pose estimation files (appended to'
             ' video file after removing extension)',
        nargs='+',
        required=True,
    )
    dir_parser.add_argument(
        '--num-procs',
        help='the number of processes to use',
        default=2,
        type=int,
    )
    dir_parser.add_argument(
        '--batch-file',
        help='a newline separated list of video files to process. Paths'
             ' should be relative to the given --in-dir. The default'
             ' behavior if this argument is missing is to traverse the'
             ' --in-dir and process all AVI files.',
        required=False,
    )

    vid_parser = subparsers.add_parser(
        'vid',
        help='vid subcommand help (for processing a single video)')
    vid_parser.set_defaults(subcommand='vid')

    vid_parser.add_argument(
        '--in-vid',
        help='input video to process',
        required=True,
    )
    vid_parser.add_argument(
        '--in-pose',
        help='input HDF5 pose file',
        required=True,
    )
    vid_parser.add_argument(
        '--out-vid',
        help='output pose overlay video to generate',
        required=True,
    )

    args = parser.parse_args()

    exclude_points = set()
    if args.exclude_forepaws:
        exclude_points.add(LEFT_FRONT_PAW_INDEX)
        exclude_points.add(RIGHT_FRONT_PAW_INDEX)
    if args.exclude_ears:
        exclude_points.add(LEFT_EAR_INDEX)
        exclude_points.add(RIGHT_EAR_INDEX)

    if 'subcommand' in args:
        if args.subcommand == 'dir':

            out_dir = args.in_dir
            if args.out_dir is not None:
                out_dir = args.out_dir

            files_to_process = []
            if args.batch_file is not None:
                with open(args.batch_file) as f:
                    for line in f:
                        files_to_process.append(line.strip())

            else:
                for dirname, _, filelist in os.walk(args.in_dir):
                    for fname in filelist:
                        if fname.lower().endswith('.avi'):
                            fpath = os.path.join(dirname, fname)
                            rel_fpath = os.path.relpath(fpath, args.in_dir)
                            files_to_process.append(rel_fpath)

            with mp.Pool(args.num_procs) as p:
                for rel_fpath in files_to_process:
                    for pose_suffix in args.pose_suffix:
                        p.apply_async(
                            process_video_relpath,
                            (rel_fpath, pose_suffix, args.in_dir, out_dir, exclude_points),
                            dict(),
                            lambda x: None,
                            lambda x: print(x))

                p.close()
                p.join()

        elif args.subcommand == 'vid':
            process_video(args.in_vid, args.in_pose, args.out_vid, exclude_points)

    else:
        print('ERROR: dir or vid subcommand must be specified')

if __name__ == '__main__':
    main()
