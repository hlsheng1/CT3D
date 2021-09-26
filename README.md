## "Improving 3D Object Detection with Channel-wise Transformer"

Thanks for the [OpenPCDet](https://github.com/open-mmlab/OpenPCDet), this implementation of the CT3D is mainly based on the pcdet v0.3. Our paper can be downloaded here [ICCV2021](https://arxiv.org/abs/2108.10723). 

![CT3D](C:\Users\poplar\Desktop\CT3D.jpg) Overview of CT3D. The raw points are first fed into the RPN for generating 3D proposals. Then the raw points along with the corresponding proposals are processed by the channel-wise Transformer composed of the proposal-to-point encoding module and the channel-wise decoding module. Specifically, the proposal-to-point encoding module is to modulate each point feature with global proposal-aware context information. After that, the encoded point features  are transformed into an effective proposal feature representation by the channel-wise decoding module for confidence prediction and box regression.

|                         | AP@R11 | AP@R40 |                           Download                           |
| :---------------------: | :----: | :----: | :----------------------------------------------------------: |
|        Only Car         | 86.06  | 85.79  | [model-car](https://drive.google.com/file/d/1ZWybnLJIzN4d8QBtctRJHd03sW51nrJa/view?usp=sharing) |
|    3-Category (Car)     | 85.04  | 84.97  | [model-3cat](https://drive.google.com/file/d/1lTSoWd4XzeG6FkDY-Sd-1_HgNgHqdHsl/view?usp=sharing) |
| 3-Category (Pedestrian) | 56.28  | 55.58  |                              -                               |
|  3-Category (Cyclist)   | 71.71  | 71.88  |                              -                               |

### 1. Recommended Environment

- Linux (tested on Ubuntu 16.04)
- Python 3.6+
- PyTorch 1.1 or higher (tested on PyTorch 1.6)
- CUDA 9.0 or higher (PyTorch 1.3+ needs CUDA 9.2+)

### 2. Set the Environment

```shell
pip install -r requirement.txt
python setup.py develop
```

### 3. Data Preparation

- Prepare [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and [road planes](https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view?usp=sharing)

```shell
# Download KITTI and organize it into the following form:
├── data
│   ├── kitti
│   │   │── ImageSets
│   │   │── training
│   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes)
│   │   │── testing
│   │   │   ├──calib & velodyne & image_2

# Generatedata infos:
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```

- Prepare [Waymo](https://waymo.com/open/download/) dataset

```shell
# Download Waymo and organize it into the following form:
├── data
│   ├── waymo
│   │   │── ImageSets
│   │   │── raw_data
│   │   │   │── segment-xxxxxxxx.tfrecord
|   |   |   |── ...
|   |   |── waymo_processed_data
│   │   │   │── segment-xxxxxxxx/
|   |   |   |── ...
│   │   │── pcdet_gt_database_train_sampled_xx/
│   │   │── pcdet_waymo_dbinfos_train_sampled_xx.pkl

# Install tf 2.1.0
# Install the official waymo-open-dataset by running the following command:
pip3 install --upgrade pip
pip3 install waymo-open-dataset-tf-2-1-0 --user

# Extract point cloud data from tfrecord and generate data infos:
python -m pcdet.datasets.waymo.waymo_dataset --func create_waymo_infos --cfg_file tools/cfgs/dataset_configs/waymo_dataset.yaml
```

### 4. Train

- Train with a single GPU

```shell
python train.py --cfg_file ${CONFIG_FILE}

# e.g.,
python train.py --cfg_file tools/cfgs/kitti_models/second_ct3d.yaml
```

- Train with multiple GPUs or multiple machines

```shell
bash scripts/dist_train.sh ${NUM_GPUS} --cfg_file ${CONFIG_FILE}
# or 
bash scripts/slurm_train.sh ${PARTITION} ${JOB_NAME} ${NUM_GPUS} --cfg_file ${CONFIG_FILE}

# e.g.,
bash scripts/dist_train.sh 8 --cfg_file tools/cfgs/kitti_models/second_ct3d.yaml
```

### 5. Test

- Test with a pretrained model:

```shell
python test.py --cfg_file ${CONFIG_FILE} --ckpt ${CKPT}

# e.g., 
python test.py --cfg_file tools/cfgs/kitti_models/second_ct3d.yaml --ckpt output/kitti_models/second_ct3d/default/kitti_val.pth
```
