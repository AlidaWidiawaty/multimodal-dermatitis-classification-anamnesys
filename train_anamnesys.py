import torch
import random
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from dataset import AnamnesysDataset
from models import AnamnesysModel


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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set deterministic behavior
set_deterministic(seed=42)

# Load the dataset
dataset = AnamnesysDataset(text_file_path)

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
model = AnamnesysModel().to(device)

# Define optimizer and loss function
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-5)
scheduler = StepLR(optimizer, step_size=2, gamma=0.1)

# Set the number of epochs and device (GPU or CPU)
num_epochs = 20

best_acc = 0

# ADDED: Initialize lists to record metrics
history = {
    "epoch": [],
    "train_loss": [],
    "val_loss": [],
    "train_accuracy": [],
    "val_accuracy": [],
    "train_f1": [],
    "val_f1": [],
}

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    all_preds_train = []
    all_labels_train = []

    # Training phase
    for batch in tqdm(
        train_dataloader,
        desc=f"Epoch {epoch + 1}/{num_epochs} - Training",
        unit="batch",
    ):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * input_ids.size(0)

        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels)
        total_predictions += labels.size(0)

        # ADDED: Collect preds and labels for F1 calculation
        all_preds_train.extend(preds.cpu().numpy())
        all_labels_train.extend(labels.cpu().numpy())

    # Calculate training loss and accuracy
    train_loss = running_loss / len(train_dataloader.dataset)
    train_accuracy = correct_predictions.double() / total_predictions
    train_f1 = f1_score(all_labels_train, all_preds_train, average="weighted")

    print(
        f"Epoch {epoch + 1}/{num_epochs} - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, F1-score: {train_f1:.4f}"
    )

    # Evaluation phase on validation set
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    val_correct_predictions = 0
    val_total_predictions = 0

    all_preds_val = []
    all_labels_val = []

    with torch.no_grad():  # Disable gradient computation during evaluation
        for batch in tqdm(
            val_dataloader,
            desc=f"Epoch {epoch + 1}/{num_epochs} - Validation",
            unit="batch",
        ):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * input_ids.size(0)

            _, preds = torch.max(outputs, 1)
            val_correct_predictions += torch.sum(preds == labels)
            val_total_predictions += labels.size(0)

            # ADDED: Collect preds and labels for F1 calculation
            all_preds_val.extend(preds.cpu().numpy())
            all_labels_val.extend(labels.cpu().numpy())

    # Calculate validation loss and accuracy
    val_loss = val_loss / len(val_dataloader.dataset)
    val_accuracy = val_correct_predictions.double() / val_total_predictions
    val_f1 = f1_score(all_labels_val, all_preds_val, average="weighted")

    print(
        f"Epoch {epoch + 1}/{num_epochs} - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation F1-score: {val_f1:.4f}"
    )

    # ADDED: Record metrics for this epoch
    history["epoch"].append(epoch + 1)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_accuracy"].append(train_accuracy.item())
    history["val_accuracy"].append(val_accuracy.item())
    history["train_f1"].append(train_f1)
    history["val_f1"].append(val_f1)

    if best_acc < val_accuracy:
        best_acc = val_accuracy
        torch.save(model.state_dict(), "anamnesys_best_model.pth")
        print("Best model saved to anamnesys_best_model.pth")

# ADDED: Save metrics to CSV
df_history = pd.DataFrame(history)
df_history.to_csv("training_history.csv", index=False)
print("Training history saved to training_history.csv")

# ADDED: Plot metrics
plt.figure(figsize=(14, 8))

plt.subplot(2, 2, 1)
plt.plot(df_history["epoch"], df_history["train_loss"], label="Train Loss")
plt.plot(df_history["epoch"], df_history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(df_history["epoch"], df_history["train_accuracy"], label="Train Accuracy")
plt.plot(df_history["epoch"], df_history["val_accuracy"], label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy over Epochs")
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(df_history["epoch"], df_history["train_f1"], label="Train F1-score")
plt.plot(df_history["epoch"], df_history["val_f1"], label="Val F1-score")
plt.xlabel("Epoch")
plt.ylabel("F1-score")
plt.title("F1-score over Epochs")
plt.legend()

plt.tight_layout()
plt.savefig("training_plots.png")
plt.show()
print("Training plots saved to training_plots.png")

# Test phase (after training completes)
model.load_state_dict(torch.load("anamnesys_best_model.pth", map_location=device))
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