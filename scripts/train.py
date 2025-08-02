import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.optim as optim

from tqdm import tqdm
import yaml
import argparse
from torch.amp import GradScaler, autocast

from data.dataset import build_dataset
from models import build_model
from utils.losses import FocalLoss
from utils.ema import update_ema_model
from utils.plot import visualize_gradcam
from utils.augment import cutmix_data


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    dataset, train_loader, val_loader = build_dataset(config)

    # Model setup
    num_classes = len(dataset.features["label"].names)
    model, ema_model = build_model(config["model_name"], num_classes, device)

    optimizer = optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2
    )
    criterion = FocalLoss(gamma=2)
    scaler = GradScaler()

    best_val_loss = float("inf")
    train_losses, val_losses = [], []
    epochs_no_improve = 0

    for epoch in range(config["epochs"]):
        model.train()
        running_loss = 0

        for inputs, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels)

            optimizer.zero_grad()
            with autocast(device_type="cuda"):
                outputs = model(inputs)
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            update_ema_model(model, ema_model)
            running_loss += loss.item()

        train_losses.append(running_loss / len(train_loader))
        scheduler.step()

        # Validation
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                with autocast(device_type="cuda"):
                    outputs = ema_model(inputs)
                    val_loss += criterion(outputs, labels).item()
                    correct += (outputs.argmax(1) == labels).sum().item()
                    total += labels.size(0)

        val_losses.append(val_loss / len(val_loader))
        acc = correct / total * 100
        print(f"✅ Epoch {epoch+1}: Train Loss={train_losses[-1]:.4f}, Val Loss={val_losses[-1]:.4f}, Val Acc={acc:.2f}%")

        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            torch.save(ema_model.state_dict(), config["save_path"])
            print("🚀 Model saved.")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config["patience"]:
                print("🛑 Early stopping.")
                break

    # Optional GradCAM
    if config.get("gradcam", False):
        class_names = dataset.features["label"].names
        target_layer = eval(config["target_layer"], {"model": ema_model})
        visualize_gradcam(ema_model, val_loader, class_names, target_layer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    main(config)
