import argparse
import cv2
import numpy as np
import os
import skimage
import xml.etree.ElementTree as ET


def parse_cvat(cvat_xml_path):
    root = ET.parse(cvat_xml_path)
    for image_elem in root.findall('./image'):
        img_name = image_elem.attrib['name']

        polylines = []
        for polyline_elem in image_elem.findall('./polyline'):
            xy_strs = [
                xy_str.split(',')
                for xy_str in polyline_elem.attrib['points'].split(';')
            ]
            assert len(xy_strs) 

            xy_points = np.array(
                [(float(x_str), float(y_str)) for x_str, y_str in xy_strs],
                dtype=np.float32,
            )

            polylines.append(xy_points)

        yield {
            'image_name': img_name,
            'polylines': polylines,
        }


def render_polyline_overlay(image, polyline, color=(255, 255, 255)):

    polyline_rounded = np.rint(polyline).astype(np.int)

    # first the outline in black
    cv2.polylines(image, [polyline_rounded], False, (0, 0, 0), 2, cv2.LINE_AA)
    for point_x, point_y in polyline:
        cv2.circle(image, (point_x, point_y), 3, (0, 0, 0), -1, cv2.LINE_AA)

    # then inner trace with color
    cv2.polylines(image, [polyline_rounded], False, color, 1, cv2.LINE_AA)
    for point_x, point_y in polyline:
        cv2.circle(image, (point_x, point_y), 2, color, -1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description='render cvat annotations')

    parser.add_argument('--cvat-files',
                        help='list of CVAT XML files to use',
                        nargs='+',
                        required=True,
                        type=str)
    parser.add_argument('--image-dir',
                        help='directory containing images',
                        required=True,
                        type=str)
    parser.add_argument('--image-out-dir',
                        help='the directory we render to',
                        required=True,
                        type=str)

    args = parser.parse_args()

    if args.image_out_dir is not None:
        os.makedirs(args.image_out_dir, exist_ok=True)

    for cvat_file in args.cvat_files:
        for image_labels in parse_cvat(cvat_file):
            image_path = os.path.join(args.image_dir, image_labels['image_name'])
            image_out_path = os.path.join(args.image_out_dir, image_labels['image_name'])
            if os.path.exists(image_path):
                # image_data_numpy = skimage.io.imread(image_path, as_gray=True)
                image_data_numpy = skimage.io.imread(image_path)
                for polyline in image_labels['polylines']:
                    render_polyline_overlay(image_data_numpy, polyline)

                skimage.io.imsave(image_out_path, image_data_numpy)


if __name__ == "__main__":
    main()
