# Pedestrian detection (YOLOv8 part)

## project structure
- `datasets/`: store datasets which will be trained
- `scripts/`: store training and test scripts
- `weights/`: store the pre-trained model and the trained model
- `outputs/`: store training logs and results

## how to use
1. activate virtual environment：`conda activate yolov8`
2. training model ：`python scripts/train.py`
3. test model  ：`python scripts/test.py`

#BEVFusion part under Linux, and its compressed files and the YOLO compressed files contain datasets exceeding 25MB, which is the maximum upload limit.
  Therefore, the relevant core files and without configuration files, etc. are provided.
