import argparse
import h5py
import numpy as np
import os
from pathlib import Path, WindowsPath
import yaml

CORNERS_SUFFIX = '_corners_v2.yaml'
ARENA_SIZE_CM = 52

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--arena-size-cm',
        type=float,
        default=52,
        help='the arena size is used to derive cm/pixel using corners files',
    )
    parser.add_argument(
        'rootdir',
        help='the root directory that we parse and add unit attributes to'
    )

    args = parser.parse_args()

    for dirpath, dirnames, filenames in os.walk(args.rootdir):
        for filename in filenames:
            if filename.endswith(CORNERS_SUFFIX):
                pose_path_exists = False
                for pose_version in (4, 3, 2):
                    pose_suffix = f'_pose_est_v{pose_version}.h5'
                    pose_filename = filename[:-len(CORNERS_SUFFIX)] + pose_suffix
                    pose_path = Path(dirpath, pose_filename)
                    pose_path_exists = pose_path.exists()
                    if pose_path_exists:
                        break
                
                if pose_path_exists:
                    corners_path = Path(dirpath, filename)
                    with open(corners_path) as corners_file:
                        corners_dict = yaml.safe_load(corners_file)
                        print(list(corners_dict.keys()))
                        xs = corners_dict['corner_coords']['xs']
                        ys = corners_dict['corner_coords']['ys']

                        # get all of the non-diagonal pixel distances between
                        # corners and take the meadian
                        xy_ul, xy_ll, xy_ur, xy_lr = [
                            np.array(xy, dtype=np.float) for xy in zip(xs, ys)
                        ]
                        med_corner_dist_px = np.median([
                            np.linalg.norm(xy_ul - xy_ll),
                            np.linalg.norm(xy_ll - xy_lr),
                            np.linalg.norm(xy_lr - xy_ur),
                            np.linalg.norm(xy_ur - xy_ul),
                        ])

                        cm_per_pixel = np.float32(args.arena_size_cm / med_corner_dist_px)
                        with h5py.File(pose_path, 'r+') as pose_h5_file:
                            pose_h5_file['poseest'].attrs['cm_per_pixel'] = cm_per_pixel
                            pose_h5_file['poseest'].attrs['cm_per_pixel_source'] = 'corner_detection'


if __name__ == '__main__':
    main()
