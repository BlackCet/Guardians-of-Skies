import os
import torch
from ultralytics import YOLO

DATASET_YAML = "/kaggle/working/crop_classification/dataset.yaml"  
CHECKPOINT_DIR = "/kaggle/working/yolo-checkpoints2"  

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

num_gpus = torch.cuda.device_count()
print(f" Using {num_gpus} GPUs!")

model.train(
    data=DATASET_YAML,
    epochs=10,
    imgsz=640,
    device=0, 
    batch=16,
    save=True,
    save_period=5,
    freeze=20,
    project=CHECKPOINT_DIR,
    name="yolo_aircraft_training"
)

print(f" Checkpoints are saved in {CHECKPOINT_DIR}")
