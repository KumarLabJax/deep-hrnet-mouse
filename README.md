# Mouse Pose HR Net

This repository is a forked and significantly modified version of the [official HRNet repository](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) in order to support mouse pose inference. This repository contains two main approaches for inferring mouse pose. Firstly we have a single mouse pose inference script: `tools/infermousepose.py`. The pose output for single mouse pose is currently used by our [gait analysis system](https://github.com/KumarLabJax/gaitanalysis) to extract the gait metrics that analize. We also have implemented multi-mouse pose estimation. The entry point for multi-mouse pose is the `tools/infermultimousepose.py` script. We describe these tools and others in more detail below:

## Tools

* `tools/addpixelunits.py`: adds pixel-to-centimeter conversion metadata to the pose file
* `tools/infercorners.py`: this script performs corner detection which we use to convert pixel space to real-world physical space
* `tools/inferfecalbolicount.py`: this script will locate fecal boli in the open field and provide minute by minute counts in the output
* `tools/infermousepose.py`: we use this script to infer single mouse pose for every frame in a video
* `tools/infermultimousepose.py`: we use this script to infer multi mouse pose for every frame in a video
* `tools/mousetrain.py`: our script for training the neural network on single mouse pose
* `tools/testcornermodel.py`: script for testing the corner detection model and printing accuracy statistics
* `tools/testfecalboli.py`: script for testing the fecal boli detection model and printing accuracy statistics
* `tools/testmouseposemodel.py`: script for testing the single mouse pose model and printing accuracy statistics
* `tools/testmultimouseinference.py`: script for testing the multi-mouse model and printing accuracy statistics

This repository includes the following tools. All of them provide command line help if they are run like: `python3 tools/scriptname.py --help`. Additionally most include comments in the script source code which show example invokations.

## Installation

Before starting make sure you have `python3` installed. This code has been developed and tested on `python 3.8.10`. The recommended approach to installing dependencies is to use a virtual like:

    python3 -m venv mousepose-venv
    source mousepose-venv/bin/activate

    # now switch to the pose repo dir and install requirements
    cd $MOUSEPOSE_REPO_DIR
    pip3 install -r requirements.txt

Note that we also have prebuilt singularity VMs that we will be providing to simplify this process.

## Pose File Formats

The following describes our pose file HDF5 formats. `tools/infermousepose.py` will generate the v2 format for single mouse and `tools/infermultimousepose.py` will generate the v3 format.

### Single-Mouse Pose Estimation v2 Format

Each video has a corresponding HDF5 file that contains pose estimation coordinates and confidences. These files will have the same name as the corresponding video except that you replace ".avi" with "_pose_est_v2.h5"

Each HDF5 file contains two datasets:

* "poseest/points":
  this is a dataset with size (#frames x #keypoints x 2) where #keypoints is 12 following the indexing scheme shown below and the last dimension of size 2 is used hold the pixel (x, y) position of the respective frame # and keypoint #
  the datatype is a 16bit unsigned integer
* "poseest/confidence":
  this dataset has size (#frames x #keypoints) and assigns a 0-1 confidence value to each of the 12 points (sometimes the confidence goes slightly higher than 1). I tend to threshold at 0.3 as being "very low confidence". When the mouse is not in the arena almost all confidence values should be < 0.3.
  the datatype is a 32 bit floating point

The "poseest" group can have attributes attached

* "cm_per_pixel" (optional):
  defines how many centimeters a pixel of open field represents
  the datatype is 32 bit floating point scalar
* "cm_per_pixel_source" (optional):
  defines how the "cm_per_pixel" value was set. Value will be one of "corner_detection", "manually_set" or "default_alignment"
  the datatype is string scalar

The 12 point indexes have the following mapping to mouse body part:

* NOSE_INDEX = 0
* LEFT_EAR_INDEX = 1
* RIGHT_EAR_INDEX = 2
* BASE_NECK_INDEX = 3
* LEFT_FRONT_PAW_INDEX = 4
* RIGHT_FRONT_PAW_INDEX = 5
* CENTER_SPINE_INDEX = 6
* LEFT_REAR_PAW_INDEX = 7
* RIGHT_REAR_PAW_INDEX = 8
* BASE_TAIL_INDEX = 9
* MID_TAIL_INDEX = 10
* TIP_TAIL_INDEX = 11

### Multi-Mouse Pose Estimation v3 Format

Each video has a corresponding HDF5 file that contains pose estimation coordinates and confidences. These files will have the same name as the corresponding video except that you replace ".avi" with "_pose_est_v3.h5"

Several of the datasets below have a dimension of length "maximum # instances". This is because the instance count can vary over time for a video either because mice are added or removed, or because of an error in inference. Since each frame has it's own instance count you must consult the "poseest/instance_count" dataset to determine the number of valid instances per frame.

Each HDF5 file contains the following datasets:

* "poseest/points":
  this is a dataset with size (#frames x maximum # instances x #keypoints x 2) where #keypoints is 12 following the indexing scheme shown below and the last dimension of size 2 is used hold the pixel (y, x) position of the respective frame # and keypoint #
  the datatype is a 16bit unsigned integer
* "poseest/confidence":
  this dataset has size (#frames x maximum # instances x #keypoints) and assigns a confidence value to each of the 12 points. Values of 0 indicate a missing point. Anything higher than 0 indicates a valid point, so in that sense this dataset can be treated as binary.
  the datatype is a 32 bit floating point
* "poseest/instance_count":
  this dataset has size (#frames) and gives the instance count for every frame (this can change when mice are added and removed, or if inference fails for some frames)
  the datatype is an 8 bit unsigned integer
* "poseest/instance_embedding":
  Most applications can ignore this dataset. This is a dataset with size (#frames x maximum # instances x #keypoints) where #keypoints is 12 following the indexing scheme shown below. This dataset contains the instance embedding for the respective instance at the respective frame and point.
  the datatype is a 32 bit floating point
* "poseest/instance_track_id":
  this is a dataset with size (#frames x maximum # instances) and contains the instance_track_id for each instance index on a per frame basis.

The "poseest" group can have attributes attached

* "cm_per_pixel" (optional):
  defines how many centimeters a pixel of open field represents
  the datatype is 32 bit floating point scalar
* "cm_per_pixel_source" (optional):
  defines how the "cm_per_pixel" value was set. Value will be one of "corner_detection", "manually_set" or "default_alignment"
  the datatype is string scalar

The 12 keypoint indexes have the following mapping to mouse body part:

* NOSE_INDEX = 0
* LEFT_EAR_INDEX = 1
* RIGHT_EAR_INDEX = 2
* BASE_NECK_INDEX = 3
* LEFT_FRONT_PAW_INDEX = 4
* RIGHT_FRONT_PAW_INDEX = 5
* CENTER_SPINE_INDEX = 6
* LEFT_REAR_PAW_INDEX = 7
* RIGHT_REAR_PAW_INDEX = 8
* BASE_TAIL_INDEX = 9
* MID_TAIL_INDEX = 10
* TIP_TAIL_INDEX = 11

## Licensing

This code is released under MIT license.

The data produced in the associated paper used for training models are released on Zenodo under a Non-Commercial license.
