from ultralytics import YOLO
import torch

def main():
    # Using the model YOLOv8n, large models are not suitable for small classes
    model = YOLO("D:/yolov8_project/weights/yolov8n.pt")  # 替换为 yolov8m.pt

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Configure enhancement parameters in dataset.yaml (recommended) or pass through the train parameter
    train_args = {
        "data": "D:/yolov8_project/datasets/dataset.yaml",
        "epochs": 150,
        "batch": 8,
        "imgsz": 1630,
        "workers": 1,
        # "pretrained": True,  # Use pre-training weightsUse pre-training weights
        # "augment": True,  # Enable data enhancement
        # "mosaic": 0.8,
        # "fliplr": 0.5,
        # "flipud": 0.3,
        # "lr0": 0.003,
        # "lrf": 0.01,
        # "warmup_epochs": 5,
        # "warmup_momentum": 0.8,
        # "box": 0.05,
        # "cls": 0.5,
        # "dfl": 0.5,
        # "cos_lr": True,  # Enable cosine annealing learning rate
        # "amp": True,  # Mixed precision training
        # "patience": 50,  # Early stop strategy
        # "save_period": -1,  # Save the model once per training
        # "save": True,  # Save training results
        # "resume": False,  # Do not resume the last training
        # "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


    results = model.train(**train_args)


if __name__ == '__main__':
    main()