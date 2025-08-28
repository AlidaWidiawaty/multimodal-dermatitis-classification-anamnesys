import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from tqdm import tqdm  

from utils import SkinDataset, get_transforms, get_model, DATA_DIR, MODEL_DIR, NUM_FOLDS, NUM_CLASSES, EPOCHS, DEVICE, MODEL_NAMES

def train_all_models():
    dataset = SkinDataset(DATA_DIR, transform=get_transforms())
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    model_results = {}

    for model_name in MODEL_NAMES:
        print(f"\nðŸš€ Training Model: {model_name.upper()} with {NUM_FOLDS}-Fold Cross-Validation")
        
        fold_metrics = {"accuracy": [], "precision": [], "recall": [], "f1_score": []}

        for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
            print(f"\nðŸ”„ Training Fold {fold+1}/{NUM_FOLDS} for {model_name.upper()}")

            # Split dataset
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

            # Initialize model, optimizer, and loss function
            model = get_model(model_name).to(DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)
            criterion = nn.CrossEntropyLoss()

            # Early stopping setup
            best_val_acc = 0.0
            early_stopping_counter = 0
            patience = 3

            for epoch in range(EPOCHS):
                model.train()
                total_loss, correct, total = 0, 0, 0

                with tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} (Fold {fold+1}/{NUM_FOLDS})") as pbar:
                    for images, labels in pbar:
                        images, labels = images.to(DEVICE), labels.to(DEVICE)
                        optimizer.zero_grad()
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                        total_loss += loss.item()
                        correct += (outputs.argmax(dim=1) == labels).sum().item()
                        total += labels.size(0)
                        pbar.set_postfix(loss=total_loss/total, acc=correct/total)

                train_acc = correct / total if total > 0 else 0.0
                scheduler.step()

                # Validation Phase
                model.eval()
                val_correct, val_total = 0, 0
                all_preds, all_labels = [], []

                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(DEVICE), labels.to(DEVICE)
                        outputs = model(images)
                        preds = outputs.argmax(dim=1)
                        
                        val_correct += (preds == labels).sum().item()
                        val_total += labels.size(0)

                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())

                val_acc = val_correct / val_total if val_total > 0 else 0.0

                # Compute precision, recall, and F1-score
                report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
                precision = report["weighted avg"]["precision"]
                recall = report["weighted avg"]["recall"]
                f1_score = report["weighted avg"]["f1-score"]

                fold_metrics["accuracy"].append(val_acc)
                fold_metrics["precision"].append(precision)
                fold_metrics["recall"].append(recall)
                fold_metrics["f1_score"].append(f1_score)

                print(f"🚀 Fold [{fold+1}/{NUM_FOLDS}] Epoch [{epoch+1}/{EPOCHS}] Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1_score:.4f}")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    early_stopping_counter = 0
                    torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"{model_name}_fold{fold}.pth"))
                else:
                    early_stopping_counter += 1

                if early_stopping_counter >= patience:
                    print(f"⏰ Early stopping at epoch {epoch+1} for {model_name} (Fold {fold+1})")
                    break

        model_results[model_name] = {
            "accuracy": np.mean(fold_metrics["accuracy"]),
            "precision": np.mean(fold_metrics["precision"]),
            "recall": np.mean(fold_metrics["recall"]),
            "f1_score": np.mean(fold_metrics["f1_score"]),
        }

    print("\nðŸ“Š Final Model Performance (Averaged Over All Folds)")
    print("-" * 60)
    print(f"{'Model':<15}{'Accuracy':<10}{'Precision':<10}{'Recall':<10}{'F1-Score':<10}")
    print("-" * 60)
    for model, metrics in model_results.items():
        print(f"{model.upper():<15}{metrics['accuracy']:.4f}    {metrics['precision']:.4f}    {metrics['recall']:.4f}    {metrics['f1_score']:.4f}")

# ===============================
# Run the Training
# ===============================
if __name__ == "__main__":
    print("\n🚀 Starting Full Training Pipeline")
    train_all_models()
    print("✅ Training Completed!")