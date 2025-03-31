import streamlit as st
import cv2
import torch
import os
from dotenv import load_dotenv
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import gdown

load_dotenv()
model_path = "best40.pt"

if not os.path.exists(model_path):
    url = "https://drive.google.com/uc?id=1ktLDc1Of7wSh_BtFQe3vts9i4TRa8smK"
    gdown.download(url, model_path, quiet=False)

model = YOLO(model_path)

friend_classes = ["A10", "AG600", "AV8B", "An124", "An22", "An225", "An72", 
                  "B1", "B2", "B21", "B52", "Be200", "C130", "C17", "C2", "C390", "C5", 
                  "E7", "EF2000", "F117", "F14", "F15", "F16", "F22", "F35", "F18", "FA18", 
                  "TB001", "TB2", "Tornado", "Tu160","Mi28", 
                  "Mi8", "UH60", "WZ7", "Z19", "F4", "Tu22M", "Tu95", "US2", "Vulcan", "Y20", "YF23"]

enemy_classes = ["CH47", "CL415", "Ka27", "Ka52", "A400M","KAAN", "KJ600","H6", "J10", "J20", "J35", "JAS39", "JH7", "KC135", "KF21", 
                 "Mig29", "Mig31", "Mirage2000", "P3", "RQ4", "Rafale", "SR71", 
                 "Su24", "Su25", "Su34", "Su57", "Mi24", "Mi26", "AH64", "E2", "MQ9", "V22", "XB70", "U2", "JF17"]

def process_image(image):
    results = model(image)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    enemy_detected = False
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  
            class_id = int(box.cls[0].item())  
            confidence = box.conf[0].item()  
            class_name = model.names[class_id] if class_id < len(model.names) else "Unknown"
            
            if class_name in friend_classes:
                color, label_text = (0, 255, 0), "Friend"
            elif class_name in enemy_classes:
                color, label_text = (0, 0, 255), "Enemy"
                enemy_detected = True
            else:
                color, label_text = (255, 255, 255), "Unknown"
            
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
            font_scale, thickness = 0.7, 2
            label = f"{class_name} {confidence:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            text_x, text_y = max(x1, 10), max(y1 - 10, text_h + 10)
            box_coords = ((text_x, text_y - text_h - 5), (text_x + text_w, text_y + baseline))
            overlay = image.copy()
            cv2.rectangle(overlay, box_coords[0], box_coords[1], color, -1)
            cv2.addWeighted(overlay, 0.4, image, 0.6, 0, image)
            cv2.putText(image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            label_font_scale, label_thickness = 0.6, 2
            label_x, label_y = min(x2 + 10, image.shape[1] - 80), max(y1 + 20, text_h + 20)
            cv2.putText(image, label_text, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, color, label_thickness, cv2.LINE_AA)
    
    return image, enemy_detected

def display_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_bytes = BytesIO()
    plt.imsave(img_bytes, image, format='png')
    img_bytes.seek(0)
    st.image(img_bytes, use_container_width=True)

def main():
    st.title("🏺GuaRdIaNs oF ThE SkIeS")
    st.subheader("Decode the Skies Like an Ancient Pharaoh")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "tiff"])
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        processed_image, enemy_detected = process_image(image)
        display_image(processed_image)
        if enemy_detected:
            st.error("🚨 Enemy aircraft detected! 🚨", icon="⚠️")
        else:
            st.success("All clear! No enemy aircraft detected.", icon="✅")

if __name__ == "__main__":
    main()