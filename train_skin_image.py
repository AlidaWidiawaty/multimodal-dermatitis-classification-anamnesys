import torch
import random
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from tqdm import tqdm

from dataset import SkinImageDataset
from models import SkinImageModel


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
root_dir = "/home/cluster-dgx1/iros03/laras/Progress-Bulan-Maret/skin-lesion-classification/data/processed"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set deterministic behavior
set_deterministic(seed=42)

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Resize to the input size of ResNet50
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Load the dataset
dataset = SkinImageDataset(root_dir=root_dir, transform=transform)

# Split the dataset into training and validation sets
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

# Initialize the model
model = SkinImageModel(num_classes=2).to(device)

# Define optimizer and loss function
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-5)
scheduler = StepLR(optimizer, step_size=2, gamma=0.1)

num_epochs = 100  # Set number of epochs for training

best_acc = 0

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for batch in tqdm(
        train_dataloader,
        desc=f"Epoch {epoch + 1}/{num_epochs} - Training",
        unit="batch",
    ):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # Zero the gradients

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Update running loss
        running_loss += loss.item() * inputs.size(0)

        # Get accuracy
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels)
        total_predictions += labels.size(0)

    # Calculate training loss and accuracy
    train_loss = running_loss / len(train_dataloader.dataset)
    train_accuracy = correct_predictions.double() / total_predictions

    print(
        f"Epoch {epoch + 1}/{num_epochs} - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}"
    )

    # Step 6: Validation Loop with tqdm for Progress Bar
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    val_correct_predictions = 0
    val_total_predictions = 0

    # Create a tqdm progress bar for validation
    # with tqdm(
    #     val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Validation", unit="batch"
    # ) as tepoch:
    with torch.no_grad():  # No need to calculate gradients during validation
        for batch in tqdm(
            val_dataloader,
            desc=f"Epoch {epoch + 1}/{num_epochs} - Validation",
            unit="batch",
        ):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Update validation loss
            val_loss += loss.item() * inputs.size(0)

            # Get accuracy
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
    # checkpoint_filename = f"image_checkpoint_epoch_{epoch + 1}_train_acc_{train_accuracy:.4f}_val_acc_{val_accuracy:.4f}.pth"
    # torch.save(model.state_dict(), checkpoint_filename)
    # print(f"Checkpoint saved to '{checkpoint_filename}'")

    if best_acc < val_accuracy:
        best_acc = val_accuracy
        torch.save(model.state_dict(), "image_best_model.pth")
        print("Best model saved to image_best_model.pth")

# Test phase (after training completes)
model.load_state_dict(torch.load("image_best_model.pth", map_location=device))
model.to(device)
model.eval()
test_loss = 0.0
test_correct_predictions = 0
test_total_predictions = 0

with torch.no_grad():
    for batch in tqdm(
        test_dataloader,
        desc="Testing",
        unit="batch",
    ):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        test_loss += loss.item() * input_ids.size(0)

        _, preds = torch.max(outputs, 1)
        test_correct_predictions += torch.sum(preds == labels)
        test_total_predictions += labels.size(0)

test_loss = test_loss / len(test_dataloader.dataset)
test_accuracy = test_correct_predictions.double() / test_total_predictions

print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")