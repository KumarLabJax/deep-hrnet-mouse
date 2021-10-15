import argparse
import csv
import imageio
import itertools
import math
import numpy as np
import os


# Example:
#
#   share_root='/home/sheppk/smb/labshare'
#   python tools/extractframes.py \
#       --root-dir "${share_root}" \
#       --videos \
#           "${share_root}"/NV5-CBAX2/2020-04-22/MDX0159_2020-04-22_23-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-22/MDX0159_2020-04-23_11-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-22/MDX0159_2020-04-23_23-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-22/MDX0159_2020-04-24_12-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-22/MDX0159_2020-04-25_00-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-22/MDX0159_2020-04-25_11-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-22/MDX0159_2020-04-26_06-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-22/MDX0159_2020-04-26_21-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-22/MDX0159_2020-04-27_11-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-22/MDX0159_2020-04-28_01-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-22/MDX0159_2020-04-28_16-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-22/MDX0159_2020-04-29_09-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-22/MDX0159_2020-04-29_23-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-22/MDX0159_2020-04-30_11-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-22/MDX0159_2020-05-01_06-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-22/MDX0159_2020-05-02_02-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-22/MDX0159_2020-05-02_18-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-22/MDX0159_2020-05-03_07-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-22/MDX0159_2020-05-03_21-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-22/MDX0159_2020-05-04_11-00-00.avi \
#           "${share_root}"/NV5-CBAX2/2020-03-26/MDX0148_2020-03-26_17-00-00.avi "${share_root}"/NV5-CBAX2/2020-03-26/MDX0148_2020-03-27_06-00-00.avi "${share_root}"/NV5-CBAX2/2020-03-26/MDX0148_2020-03-28_00-00-00.avi "${share_root}"/NV5-CBAX2/2020-03-26/MDX0148_2020-03-28_16-00-00.avi "${share_root}"/NV5-CBAX2/2020-03-26/MDX0148_2020-03-29_06-00-00.avi "${share_root}"/NV5-CBAX2/2020-03-26/MDX0148_2020-03-29_21-00-00.avi \
#       --frame-indexes 600 \
#       --outdir fecal-boli-image-batch4

#   python tools/extractframes.py \
#       --root-dir "${share_root}" \
#       --videos \
#           "${share_root}"/NV5-CBAX2/2020-04-09/MDX0159_2020-04-09_22-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-09/MDX0159_2020-04-10_09-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-09/MDX0159_2020-04-11_01-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-09/MDX0159_2020-04-11_13-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-09/MDX0159_2020-04-12_01-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-09/MDX0159_2020-04-12_14-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-09/MDX0159_2020-04-13_01-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-09/MDX0159_2020-04-13_13-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-09/MDX0159_2020-04-14_00-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-09/MDX0159_2020-04-14_12-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-09/MDX0159_2020-04-14_23-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-09/MDX0159_2020-04-15_13-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-09/MDX0159_2020-04-16_01-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-09/MDX0159_2020-04-16_13-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-09/MDX0159_2020-04-17_03-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-09/MDX0159_2020-04-17_17-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-09/MDX0159_2020-04-18_11-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-09/MDX0159_2020-04-19_03-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-09/MDX0159_2020-04-19_14-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-09/MDX0159_2020-04-20_07-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-09/MDX0159_2020-04-21_00-00-00.avi \
#       --frame-indexes 600 \
#       --outdir fecal-boli-image-batch4


def write_frames(root_dir, net_id, frame_indexes, out_dir):
    vid_fname = os.path.join(root_dir, net_id)
    print('Processing:', vid_fname, 'with', len(frame_indexes), 'frames')

    net_id_root, _ = os.path.splitext(net_id)

    frame_indexes = sorted(frame_indexes)
    os.makedirs(out_dir, exist_ok=True)
    with imageio.get_reader(vid_fname) as reader:
        for frame_index in frame_indexes:
            if frame_index < 0:
                print('ignoring negative frame index', frame_index)
                continue

            img_data = reader.get_data(frame_index)
            frame_fname = '{}_{}.png'.format(
                net_id_root.replace('/', '+').replace('\\', '+'),
                frame_index)
            imageio.imwrite(os.path.join(out_dir, frame_fname), img_data)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--videos',
        nargs='+',
        help='the input videos',
    )
    parser.add_argument(
        '--frame-indexes',
        type=int,
        nargs='+',
        help='the frame indexes to extract',
    )
    parser.add_argument(
        '--frame-table',
        help='A tab separated file where the first column is the video NetID and all'
             ' subsequent columns are zero based frame indexes to extract. This argument'
             ' can be used instead of the videos and frame-indexes arguments.',
    )
    parser.add_argument(
        '--frame-table-row',
        help='An optional argument to specify that just a single zero-based index row'
             ' should be processed from the frame table.',
        type=int,
    )
    parser.add_argument(
        '--root-dir',
        required=True,
        help='when determining video network ID this prefix root is stripped from the video name',
    )
    parser.add_argument(
        '--outdir',
        required=True,
        help='the output directory',
    )

    args = parser.parse_args()

    root_dir = os.path.normpath(args.root_dir)

    if args.videos is not None:
        for vid_fname in args.videos:
            net_id = os.path.relpath(os.path.normpath(vid_fname), root_dir)
            write_frames(root_dir, net_id, args.frame_indexes, args.outdir)

    if args.frame_table:
        with open(args.frame_table, newline='') as frame_table_file:
            frame_table_reader = csv.reader(frame_table_file, delimiter='\t')
            for row_index, row in enumerate(frame_table_reader):
                if args.frame_table_row is None or row_index == args.frame_table_row:
                    if len(row) >= 2:
                        net_id = row[0].strip()
                        frame_indexes = sorted(int(x.strip()) for x in row[1:])
                        write_frames(root_dir, net_id, frame_indexes, args.outdir)


if __name__ == "__main__":
    main()
