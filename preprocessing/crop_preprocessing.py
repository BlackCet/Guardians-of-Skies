import os
import shutil

cropped_dataset = r"C:\arpit\project2\crop_classification"
splits = ["train", "val", "test"]

aircraft_classes = {name: idx for idx, name in enumerate(sorted(os.listdir(os.path.join(cropped_dataset, "train"))))}

for split in splits:
    split_path = os.path.join(cropped_dataset, split)
    images_path = os.path.join(split_path, "images")
    labels_path = os.path.join(split_path, "labels")

    os.makedirs(images_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)

    for aircraft_model in aircraft_classes:
        model_path = os.path.join(split_path, aircraft_model)
        if not os.path.isdir(model_path):
            continue

        for img_file in os.listdir(model_path):
            if img_file.endswith((".jpg", ".png", ".jpeg")):
                old_img_path = os.path.join(model_path, img_file)
                new_img_path = os.path.join(images_path, img_file)
                shutil.move(old_img_path, new_img_path)

                class_id = aircraft_classes[aircraft_model]
                label_file = os.path.join(labels_path, img_file.rsplit(".", 1)[0] + ".txt")
                with open(label_file, "w") as f:
                    f.write(f"{class_id} 0.5 0.5 1.0 1.0\n") 

        shutil.rmtree(model_path)

print("✅ Cropped dataset restructured successfully!")
