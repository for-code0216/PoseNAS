# Pose-native Neural Architecture Search for Multi-person Pose Estimation

## Introduction
This is the pytorch implementation of <u>Pose-native Neural Architecture Search for Multi-person Pose Estimation</u>. 

In this work, we present the Pose-native Network Architecture Search (PoseNAS) to simultaneously design a pose encoder and pose decoder for pose estimation. Specifically, we directly search a data-oriented pose encoder with stacked searchable cells, which can provide an optimum feature extractor for the pose specific task. In the pose decoder, we exploit scale-adaptive fusion cells to promote rich information exchange across the multi-scale feature maps. Meanwhile, the pose decoder adopts a Fusion-and-Enhancement manner to progressively boost the high-resolution representations that are non-trivial for the precious prediction of hard keypoints. With the exquisitely designed search space and search strategy, PoseNAS can simultaneously search all modules in an end-to-end manner. Our best model obtains $76.7\%$ mAP and  $75.9\%$ mAP on the COCO validation set and test set with only $33.6$M parameters. 

## Architecture Description
In this project, we support two different structures, PoseNAS-L18-C48 and PoseNAS-L18-C64. 'L' and 'C' stand for the number of  cells in the pose encoder and the number of the initial channels of the pose encoder, respectively, you can change them in the Configuration files (config.TRAIN.LAYERS and config.TRAIN.INIT_CHANNELS). 

## Main Results
### Results on MPII val
| Arch            | Head | Shoulder | Elbow | Wrist | Hip  | Knee | Ankle | Mean |
| --------------- | ---- | -------- | ----- | ----- | ---- | ---- | ----- | ---- |
| PoseNAS-L18-C64 | 97.2 | 96.3     | 90.6  | 86.0  | 90.0 | 86.5 | 83.0  | 90.4 |

- Flip test is used.
- Input size is 256x256

### Results on COCO val2017 
| Arch            | Input size | #Params | GFLOPs | AP    | Ap .5 | AP .75 | AP (M) | AP (L) | AR    | AR .5 | AR .75 | AR (M) | AR (L) |
| --------------- | ---------- | ------- | ------ | ----- | ----- | ------ | ------ | ------ | ----- | ----- | ------ | ------ | ------ |
| PoseNAS-L18-C48 | 384x288    | 21.1M   | 9.1    | 0.762 | 0.910 | 0.830  | 0.723  | 0.828  | 0.810 | 0.944 | 0.871  | 0.768  | 0.871  |
| PoseNAS-L18-C64 | 384x288    | 33.6M   | 14.8   | 0.767 | 0.915 | 0.837  | 0.725  | 0.829  | 0.812 | 0.947 | 0.873  | 0.771  | 0.871  |

- Flip test is used.
- Person detector is available at [detected_bbox](https://).
- GFLOPs is for convolution and linear layers only.


### Results on COCO test-dev2017 
| Arch            | Input size | #Params | GFLOPs | AP    | Ap .5 | AP .75 | AP (M) | AP (L) | AR    | AR .5 | AR .75 | AR (M) | AR (L) |
| --------------- | ---------- | ------- | ------ | ----- | ----- | ------ | ------ | ------ | ----- | ----- | ------ | ------ | ------ |
| PoseNAS-L18-C48 | 384x288    | 21.1M   | 9.1    | 0.753 | 0.927 | 0.832  | 0.717  | 0.810  | 0.802 | 0.956 | 0.871  | 0.762  | 0.857  |
| PoseNAS-L18-C64 | 384x288    | 33.6M   | 14.8   | 0.759 | 0.930 | 0.838  | 0.722  | 0.814  | 0.807 | 0.958 | 0.876  | 0.767  | 0.861  |

- Flip test is used.
- Person detector is available at[detected_bbox](https://).
- GFLOPs is for convolution and linear layers only.


## Code description
- Currently we release our searched network model and the training code. 
- The search code will be released after the paper accepted. 


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
   # Install 
   python3 setup.py install --user
   # Alternatively, Install into global site-packages
   make install
   ```
6. Init output (training model output directory) and log (tensorboard log directory) directory:

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
**For MPII data**, please download from [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/). The original annotation files are in matlab format. We use the converted  json format provided by  [SimpleBaseline](https://github.com/microsoft/human-pose-estimation.pytorch) .
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

**For COCO data**, please download from [COCO download](http://cocodataset.org/#download) . The  person detection results of COCO val2017 and test-dev2017 are available at   [detected_bbox](https://).
Download and extract them under {POSE_ROOT}/data, and make them look like this:

```
${POSE_ROOT}
|-- data
`-- |-- coco
    `-- |-- annotations
        |   |-- person_keypoints_train2017.json
        |   `-- person_keypoints_val2017.json
        |-- person_detection_results
        |   |-- COCO_val2017_detections_bbox.json
        |   |-- COCO_test-dev2017_detections_bbox.json
        `-- images
            |-- train2017
            |   |-- 000000000009.jpg
            |   |-- 000000000025.jpg
            |   |-- ... 
            `-- val2017
                |-- 000000000139.jpg
                |-- 000000000285.jpg
                |-- ... 
```

### Training and Testing

#### Testing on COCO val2017 dataset

##### Testing on COCO dataset  using  our [pretrained model](https://drive.google.com/drive/folders/1Ydy7JXn1AAvx7av2Sc2CMl7ijJrOIWxW?usp=sharing).

```
python test.py --cfg experiments/coco/256x192_18l_64c.yaml --test_weight "path/to/your/weight"
```

#### Training on COCO train2017 dataset

```
python train.py --cfg experiments/coco/256x192_18l_64c.yaml
```



#### Testing on MPII dataset

##### Testing on MPII dataset  using  our [pretrained model](https://drive.google.com/drive/folders/1Ydy7JXn1AAvx7av2Sc2CMl7ijJrOIWxW?usp=sharing).

```
python test.py --cfg experiments/mpii/256x256_18l_64c.yaml --test_weight "path/to/your/weight"
```

#### Training on MPII dataset
```
python train.py --cfg experiments/mpii/256x256_18l_64c.yaml
```



## Acknowledgements
This repo is largely modified from [DARTS](https://github.com/quark0/darts), [HrNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) and [Simple Baseline](https://github.com/microsoft/human-pose-estimation.pytorch).

## Reference
  ```
@InProceedings{AllSearchable,
  author = {Anonymous ACM MM submission: Paper ID 795},
  title = {APose-native Neural Architecture Search for Multi-person Pose Estimation},
  booktitle = {Submitted to ACM International Conference on Multimedia (ACM MM)},
  year = {2020}
}

  ```
