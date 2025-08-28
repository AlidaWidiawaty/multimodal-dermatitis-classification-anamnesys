import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import random
import numpy as np

from dataset import MultimodalDataset
from models import MultimodalModel


# Set deterministic behavior
def set_deterministic(seed=42):
    # Set the random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure that CUDA operations are deterministic (may slow down performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = (
        False  # Disable the autotuner for optimal algorithms
    )
    torch.backends.cudnn.enabled = True


# File paths and device setup
text_file_path = "preprocessed_dataset.csv"
image_dir = "dataset-new/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set deterministic behavior
set_deterministic(seed=42)

# Data transformation for the images (resize, normalize, etc.)
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Resize image to 224x224
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # Normalize image
    ]
)

# Create dataset using the MultimodalDataset class
dataset = MultimodalDataset(
    text_file_path=text_file_path,
    image_dir=image_dir,
    max_length=512,
    transform=transform,
)

# Split dataset into training (80%), validation (10%), and test (10%) datasets
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)

# DataLoader setup for batching and shuffling data
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Model setup: MultimodalModel with 2 output classes (binary classification)
model = MultimodalModel(num_classes=2).to(device)

# Loss function and optimizer setup
criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=2, gamma=0.1)

# Number of epochs to train the model
num_epochs = 100

best_acc = 0

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    # Training phase
    for batch in tqdm(
        train_dataloader,
        desc=f"Epoch {epoch + 1}/{num_epochs} - Training",
        unit="batch",
    ):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        images = batch["image"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()  # Clear the gradients from the previous step

        # Forward pass through the model
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, image=images
        )

        # Compute the loss
        loss = criterion(outputs, labels)
        loss.backward()  # Backpropagate the gradients
        optimizer.step()  # Update the model parameters

        running_loss += loss.item() * input_ids.size(0)  # Accumulate loss

        # Calculate number of correct predictions
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels)
        total_predictions += labels.size(0)

    # Calculate training loss and accuracy
    train_loss = running_loss / len(train_dataloader.dataset)
    train_accuracy = correct_predictions.double() / total_predictions

    print(
        f"Epoch {epoch + 1}/{num_epochs} - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}"
    )

    # Validation phase
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    val_correct_predictions = 0
    val_total_predictions = 0

    with torch.no_grad():  # Disable gradient computation for validation phase
        for batch in tqdm(
            val_dataloader,
            desc=f"Epoch {epoch + 1}/{num_epochs} - Validation",
            unit="batch",
        ):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images = batch["image"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, image=images
            )
            loss = criterion(outputs, labels)

            val_loss += loss.item() * input_ids.size(0)  # Accumulate validation loss

            # Calculate number of correct predictions for validation
            _, preds = torch.max(outputs, 1)
            val_correct_predictions += torch.sum(preds == labels)
            val_total_predictions += labels.size(0)

    # Calculate validation loss and accuracy
    val_loss = val_loss / len(val_dataloader.dataset)
    val_accuracy = val_correct_predictions.double() / val_total_predictions

    print(
        f"Epoch {epoch + 1}/{num_epochs} - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}"
    )

    # Save the model checkpoint with epoch number, training accuracy, and validation accuracy
    # checkpoint_filename = f"checkpoint_epoch_{epoch + 1}_train_acc_{train_accuracy:.4f}_val_acc_{val_accuracy:.4f}.pth"
    # torch.save(model.state_dict(), checkpoint_filename)
    # print(f"Checkpoint saved to '{checkpoint_filename}'")

    if best_acc < val_accuracy:
        best_acc = val_accuracy
        torch.save(model.state_dict(), "multimodal_best_model.pth")
        print("Best model saved to multimodal_best_model.pth")

# Test phase (after training completes)
model.load_state_dict(torch.load("multimodal_best_model.pth", map_location=device))
model.to(device)
model.eval()  # Set model to evaluation mode for testing
test_loss = 0.0
test_correct_predictions = 0
test_total_predictions = 0

with torch.no_grad():  # Disable gradient computation for testing phase
    for batch in tqdm(
        test_dataloader,
        desc="Testing",
        unit="batch",
    ):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        images = batch["image"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask, image=images)
        loss = criterion(logits, labels)

        test_loss += loss.item() * input_ids.size(0)  # Accumulate test loss

        # Calculate number of correct predictions for test
        _, preds = torch.max(logits, 1)
        test_correct_predictions += torch.sum(preds == labels.data)
        test_total_predictions += labels.size(0)

# Calculate test loss and accuracy
test_loss = test_loss / len(test_dataloader.dataset)
test_accuracy = test_correct_predictions.double() / test_total_predictions

print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")