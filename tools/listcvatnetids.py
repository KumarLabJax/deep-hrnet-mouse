import argparse
import re

import csv
import functools
import itertools
import os
import pprint
import random
import shutil

import _init_paths
from dataset.multimousepose import parse_poses


def parse_args():
    parser = argparse.ArgumentParser(
        description='list all of the vidoe netids used for the cvat annotations',
    )

    parser.add_argument('--cvat-files',
                        help='list of CVAT XML files',
                        nargs='+',
                        required=True,
                        type=str)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # Data loading code
    img_pat = re.compile(r'(.+\.avi)_([0-9]+)\.png')
    pose_labels = list(itertools.chain.from_iterable(parse_poses(f) for f in args.cvat_files))
    for pose_label in pose_labels:
        img_name = pose_label['image_name']

        m = img_pat.match(pose_label['image_name'])
        assert m

        print(m.group(1).replace('+', '/'))


if __name__ == '__main__':
    main()
