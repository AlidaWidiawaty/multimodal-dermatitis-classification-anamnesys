import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import os

INPUT_DIR = os.environ.get("INPUT_DIR", "/default/fallback/path")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/default/fallback/path")
NUM_OF_AUGMENTATIONS = 5

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=30, p=0.7),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),  # Mild Gaussian Blur
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),  # Low Gaussian Noise
])

def augment_images():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for label in os.listdir(INPUT_DIR):
        label_dir = os.path.join(INPUT_DIR, label)
        if not os.path.isdir(label_dir):
            continue

        save_label_dir = os.path.join(OUTPUT_DIR, label)
        os.makedirs(save_label_dir, exist_ok=True)

        for img_name in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_name)
            img = cv2.imread(img_path)

            if img is None:
                print(f"⚠️ Skipping file {img_path} (not a valid image)")
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            
            for i in range(NUM_OF_AUGMENTATIONS):
                augmented = transform(image=img)["image"]  
                
                aug_img = np.clip(augmented, 0, 255).astype(np.uint8)

                aug_img_path = os.path.join(save_label_dir, f"{img_name.split('.')[0]}_aug_{i}.jpg")
                cv2.imwrite(aug_img_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))  # Convert back to BGR

    print("✅ Data Augmentation Completed!")

if __name__ == "__main__":
    augment_images()