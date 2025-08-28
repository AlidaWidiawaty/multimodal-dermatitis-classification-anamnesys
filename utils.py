import os
import torch
import timm
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

from transformers import AutoModelForImageClassification
from torch.utils.data import Dataset
from PIL import Image

DATA_DIR = os.environ.get("DATA_DIR", "/default/fallback/path")
MODEL_DIR = os.environ.get("MODEL_DIR", "/default/fallback/path")
NUM_FOLDS = 10  
NUM_CLASSES = 2  
EPOCHS = 10  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAMES = ["resnet", "vit", "efficientnet"]

class SkinDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images, self.labels = [], []
        self.label_to_idx = {"DA": 0, "Non-DA": 1}  # Assign class indices explicitly

        for label in ["DA", "Non-DA"]:  # Iterate through fixed class names
            label_dir = os.path.join(data_dir, label)
            if os.path.isdir(label_dir):
                for img_name in os.listdir(label_dir):
                    img_path = os.path.join(label_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(self.label_to_idx[label])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_model(model_type):
    if model_type == "resnet":
        model = timm.create_model("resnet50", pretrained=True, num_classes=NUM_CLASSES)
    elif model_type == "vit":
        # Create a wrapper class for the ViT model
        class ViTWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = AutoModelForImageClassification.from_pretrained(
                    "google/vit-base-patch16-224-in21k", 
                    num_labels=NUM_CLASSES
                )
                
            def forward(self, x):
                return self.model(x).logits
                
        model = ViTWrapper()
    elif model_type == "efficientnet":
        model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=NUM_CLASSES)
    else:
        raise ValueError("Invalid model type")

    return model.to(DEVICE)