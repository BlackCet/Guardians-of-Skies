import os
import cv2
import torch
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
from ultralytics import YOLO
from glob import glob


DATASET_DIR = "/kaggle/working/dataset"
SAM_CHECKPOINT = "/kaggle/working/sam_vit_h.pth" 
MODEL_TYPE = "vit_h" 


device = "cuda" if torch.cuda.is_available() else "cpu"


sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT).to(device)
predictor = SamPredictor(sam)

def get_yolo_bbox(mask):
   
    y_indices, x_indices = np.where(mask > 0)  
    if len(y_indices) == 0 or len(x_indices) == 0:
        return None  
    
    
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()

   
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min
    
    return cx, cy, w, h

def process_split(split_name):
    """Process images in the given dataset split (train, valid, or test)."""
    images_path = os.path.join(DATASET_DIR, split_name, "images")  
    labels_path = os.path.join(DATASET_DIR, split_name, "labels")  
    
    os.makedirs(labels_path, exist_ok=True)  

  
    image_files = glob(os.path.join(images_path, "*.jpg")) + glob(os.path.join(images_path, "*.png"))

    for image_file in image_files:
        img = cv2.imread(image_file)  
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
        predictor.set_image(img_rgb)  

        
        masks, _, _ = predictor.predict(point_coords=None, point_labels=None, multimask_output=True)
        
       
        label_file = os.path.join(labels_path, os.path.basename(image_file).replace(".jpg", ".txt").replace(".png", ".txt"))
        
        if os.path.exists(label_file):
            os.remove(label_file)  

        with open(label_file, "w") as f:
            for mask in masks:
                bbox = get_yolo_bbox(mask)  
                if bbox:
                    cx, cy, w, h = bbox
                   
                    cx /= img.shape[1] 
                    cy /= img.shape[0] 
                    w /= img.shape[1] 
                    h /= img.shape[0] 

                    class_id = 0  
                    f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        
        print(f"Processed {image_file}") 


for split in ["train", "valid", "test"]:
    print(f"Processing {split} set...")
    process_split(split)
    
print("✅ SAM segmentation complete! YOLO labels updated.")  
