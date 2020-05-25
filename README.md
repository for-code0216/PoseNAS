# All-Searchable: Full Neural Architecture Search for Human Pose Estimation

## Introduction
This is an official pytorch implementation of [*Deep High-Resolution Representation Learning for Human Pose Estimation*](https://arxiv.org/abs/1902.09212). 
In this work, we are interested in the human pose estimation problem with a focus on learning reliable high-resolution representations. Most existing methods **recover high-resolution representations from low-resolution representations** produced by a high-to-low resolution network. Instead, our proposed network **maintains high-resolution representations** through the whole process.
We start from a high-resolution subnetwork as the first stage, gradually add high-to-low resolution subnetworks one by one to form more stages, and connect the mutli-resolution subnetworks **in parallel**. We conduct **repeated multi-scale fusions** such that each of the high-to-low resolution representations receives information from other parallel representations over and over, leading to rich high-resolution representations. As a result, the predicted keypoint heatmap is potentially more accurate and spatially more precise. We empirically demonstrate the effectiveness of our network through the superior pose estimation results over two benchmark datasets: the COCO keypoint detection dataset and the MPII Human Pose dataset. </br>

## Architecture Description
In this work, we only support three different structures(AutoPose-L12-C32,AutoPose-L18-C48,AutoPose-L18-C64), 'L' and 'C' stand for the number of the cells and the number of the initial channels of
the backbone 

## Main Results
### Results on MPII val
| Arch               | Head | Shoulder | Elbow | Wrist |  Hip | Knee | Ankle | Mean | Mean@0.1 |
|--------------------|------|----------|-------|-------|------|------|-------|------|----------|
| AutoPose-L12-C32   | 96.0 |     94.4 |  86.1 |  80.3 | 87.2 | 81.2 |  77.0 | 86.7 |     30.3 |
| AutoPose-L18-C48   | 96.6 |     95.9 |  89.4 |  84.2 | 88.7 | 85.6 |  80.9 | 89.3 |     34.9 |
| AutoPose-L18-C64   | 96.7 |     95.7 |  89.6 |  84.5 | 89.2 | 85.9 |  81.3 | 89.5 |     35.2 |


### Note:
- Flip test is used.
- Input size is 256x256

### Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset
| Arch               | Input size | #Params | GFLOPs |    AP | Ap .5 | AP .75 | AP (M) | AP (L) |    AR | AR .5 | AR .75 | AR (M) | AR (L) |
|--------------------|------------|---------|--------|-------|-------|--------|--------|--------|-------|-------|--------|--------|--------|
| AutoPose-L12-C32   |    256x192 | 5.25M   |   0.99 | 0.673 | 0.880 | 0.752  | 0.643  | 0.735  | 0.734 | 0.921 | 0.806  | 0.695  | 0.790  |
| AutoPose-L12-C32   |    384x288 | 5.25M   |   2.22 | 0.711 | 0.888 | 0.786  | 0.675  | 0.779  | 0.765 | 0.928 | 0.833  | 0.722  | 0.828  |
| AutoPose-L18-C48   |    256x192 | 15.0M   |   3.00 | 0.716 | 0.892 | 0.796  | 0.683  | 0.780  | 0.772 | 0.932 | 0.844  | 0.733  | 0.829  |
| AutoPose-L18-C48   |    384x288 | 15.0M   |   5.33 | 0.742 | 0.899 | 0.812  | 0.704  | 0.812  | 0.794 | 0.937 | 0.856  | 0.751  | 0.856  |
| AutoPose-L18-C64   |    256x192 | 26.6M   |   6.75 | 0.728 | 0.896 | 0.806  | 0.694  | 0.792  | 0.783 | 0.935 | 0.851  | 0.742  | 0.840  |
| AutoPose-L18-C64   |    384x288 | 26.6M   |   12.0 | 0.753 | 0.906 | 0.820  | 0.716  | 0.821  | 0.803 | 0.942 | 0.863  | 0.760  | 0.866  |


### Note:
- Flip test is used.
- Person detector has person AP of 56.4 on COCO val2017 dataset.
- GFLOPs is for convolution and linear layers only.


### Results on COCO test-dev2017 with detector having human AP of 60.9 on COCO test-dev2017 dataset
| Arch               | Input size | #Params | GFLOPs |    AP | Ap .5 | AP .75 | AP (M) | AP (L) |    AR | AR .5 | AR .75 | AR (M) | AR (L) |
|--------------------|------------|---------|--------|-------|-------|--------|--------|--------|-------|-------|--------|--------|--------|
| AutoPose-L12-C32   |    384x288 | 5.25M   |   2.22 | 0.703 | 0.910 |  0.781 |  0.669 |  0.762 | 0.758 | 0.944 |  0.829 |  0.716 |  0.816 |
| AutoPose-L18-C48   |    384x288 | 15.0M   |   5.33 | 0.734 | 0.921 |  0.813 |  0.699 |  0.793 | 0.786 | 0.952 |  0.856 |  0.744 |  0.843 |
| AutoPose-L18-C64   |    384x288 | 26.6M   |   12.0 | 0.744 | 0.923 |  0.824 |  0.709 |  0.803 | 0.795 | 0.955 |  0.867 |  0.755 |  0.851 |

### Note:
- Flip test is used.
- Person detector has person AP of 56.4 on COCO val2017 dataset.
- GFLOPs is for convolution and linear layers only.

## Quick start
### Installation
1. Install pytorch >= v1.0.0 following [official instruction](https://pytorch.org/).
2. Clone this repo, and we'll call the directory that you cloned as ${POSE_ROOT}.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Make libs:
   ```
   cd ${POSE_ROOT}
   make
   ```
5. Install [COCOAPI](https://github.com/cocodataset/cocoapi):
   ```
   # COCOAPI=/path/to/clone/cocoapi
   git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
   cd $COCOAPI/PythonAPI
   # Install into global site-packages
   make install
   # Alternatively, if you do not have permissions or prefer
   # not to install the COCO API into global site-packages
   python3 setup.py install --user
   ```
   Note that instructions like # COCOAPI=/path/to/install/cocoapi indicate that you should pick a path where you'd like to have the software cloned and then set an environment variable (COCOAPI in this case) accordingly.
6. Init output(training model output directory) and log(tensorboard log directory) directory:

   ```
   mkdir output 
   mkdir log
   ```

   Your directory tree should look like this:

   ```
   ${POSE_ROOT}
   ├── data
   ├── core
   ├── dataset
   ├── experiments
   ├── log
   ├── models
   ├── nms
   ├── output
   ├── utils
   ├── train.py
   ├── test.py
   ├── Makefile
   ├── README.md
   └── requirements.txt
   ```
 
### Data preparation
**For MPII data**, please download from [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/). The original annotation files are in matlab format. We have converted them into json format, you also need to download them from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blW00SqrairNetmeVu4) or [GoogleDrive](https://drive.google.com/drive/folders/1En_VqmStnsXMdldXA6qpqEyDQulnmS3a?usp=sharing).
Extract them under {POSE_ROOT}/data, and make them look like this:
```
${POSE_ROOT}
|-- data
`-- |-- mpii
    `-- |-- annot
        |   |-- gt_valid.mat
        |   |-- test.json
        |   |-- train.json
        |   |-- trainval.json
        |   `-- valid.json
        `-- images
            |-- 000001163.jpg
            |-- 000003072.jpg
```

**For COCO data**, please download from [COCO download](http://cocodataset.org/#download), 2017 Train/Val is needed for COCO keypoints training and validation. We also provide person detection result of COCO val2017 and test-dev2017 to reproduce our multi-person pose estimation results. Please download from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blWzzDXoz5BeFl8sWM-) or [GoogleDrive](https://drive.google.com/drive/folders/1fRUDNUDxe9fjqcRZ2bnF_TKMlO0nB_dk?usp=sharing).
Download and extract them under {POSE_ROOT}/data, and make them look like this:
```
${POSE_ROOT}
|-- data
`-- |-- coco
    `-- |-- annotations
        |   |-- person_keypoints_train2017.json
        |   `-- person_keypoints_val2017.json
        |-- person_detection_results
        |   |-- COCO_val2017_detections_AP_H_56_person.json
        |   |-- COCO_test-dev2017_detections_AP_H_609_person.json
        `-- images
            |-- train2017
            |   |-- 000000000009.jpg
            |   |-- 000000000025.jpg
            |   |-- 000000000030.jpg
            |   |-- ... 
            `-- val2017
                |-- 000000000139.jpg
                |-- 000000000285.jpg
                |-- 000000000632.jpg
                |-- ... 
```

### Training and Testing
#### Training on MPII dataset

```
python train.py --cfg experiments/mpii/256x256_d256x3_adam_lr1e-3.yaml
```

#### Training on COCO train2017 dataset

```
python train.py --cfg experiments/coco/256x192_d256x3_adam_lr1e-3.yaml
```
