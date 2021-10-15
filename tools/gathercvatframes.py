import argparse
import imageio
import itertools
import os
import re
import time

import _init_paths
from dataset.multimousepose import parse_poses


def _decompose_name(frame_filename):
    m = re.match(r'(.+)_([0-9]+).png', frame_filename)
    return m.group(1), int(m.group(2))

# Example call:
#
# python -u tools/gathercvatframes.py \
#   --cvat-xml data/multi-mouse/Annotations/*.xml data/multi-mouse/Annotations_NoMarkings/*.xml \
#   --outdir data/multi-mouse/Dataset \
#   --root-dir '/run/user/1002/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar' \
#   --include-neighbor-frames \
#   --vid-path-str-replace NV6-B2B NV6-CBAX2
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--cvat-xml',
        required=True,
        nargs='+',
        help='CVAT XML files that we gather frames for',
    )
    parser.add_argument(
        '--root-dir',
        required=True,
        help='the root directory location where video files are organized according to "network ID"'
    )
    parser.add_argument(
        '--include-neighbor-frames',
        action='store_true',
        help='gather neighboring frames too (ie for frame n we also save frame n-1 and n+1)'
    )
    parser.add_argument(
        '--outdir',
        required=True,
        help='the output directory',
    )
    parser.add_argument(
        '--vid-path-str-replace',
        nargs='+',
        default=[],
        help='find and replace string pairs (so this should have an even number of strings)',
    )

    args = parser.parse_args()

    all_pose_labels = itertools.chain.from_iterable(parse_poses(xml) for xml in args.cvat_xml)
    all_filenames = {lbl['image_name'] for lbl in all_pose_labels}

    assert len(args.vid_path_str_replace) % 2 == 0

    if args.include_neighbor_frames:
        for fname in set(all_filenames):
            vid_fragment, frame_index = _decompose_name(fname)
            if frame_index > 0:
                all_filenames.add('{}_{}.png'.format(vid_fragment, frame_index - 1))
            all_filenames.add('{}_{}.png'.format(vid_fragment, frame_index + 1))

    all_filenames = sorted(all_filenames, key=_decompose_name)

    for vid_frag, name_grp in itertools.groupby(all_filenames, key=lambda f: _decompose_name(f)[0]):
        missing_fnames = []
        for fname in name_grp:
            cache_file = os.path.join(args.outdir, fname)
            if os.path.exists(cache_file):
                print('EXISTS: ', cache_file)
            else:
                print('MISSING:', cache_file)
                missing_fnames.append(fname)

        if missing_fnames:
            network_filename = vid_frag.replace('+', '/')
            for i in range(len(args.vid_path_str_replace) // 2):
                find_str = args.vid_path_str_replace[i * 2]
                replace_str = args.vid_path_str_replace[i * 2 + 1]
                network_filename = network_filename.replace(find_str, replace_str)

            vid_fname = os.path.join(args.root_dir, network_filename)

            print("OPENING")
            try:
                with imageio.get_reader(vid_fname) as reader:
                    print("OPENED")
                    for fname in missing_fnames:
                        _, frame_index = _decompose_name(fname)
                        try:
                            print("GETTING")
                            img_data = reader.get_data(frame_index)
                            print("GOT")
                            imageio.imwrite(os.path.join(args.outdir, fname), img_data)
                            print("WRITTEN")

                            # sadly this sleep is needed to prevent ffmpeg from hanging
                            time.sleep(1)

                        except IndexError:
                            print('FAILED TO READ FRAME', frame_index, 'FROM', vid_fname)
            except:
                print('Failed to read video:', vid_fname)



if __name__ == "__main__":
    main()
