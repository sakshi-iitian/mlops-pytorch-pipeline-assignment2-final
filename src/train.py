import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from dataset import get_dataloaders
from model import get_model


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = correct = total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        correct += outputs.argmax(dim=1).eq(targets).sum().item()
        total += targets.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        correct += outputs.argmax(dim=1).eq(targets).sum().item()
        total += targets.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="/app/configs/training_config.yaml",
        help="Path to YAML configuration",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        config_path = Path("configs/training_config.yaml")

    config = load_config(str(config_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(
        config["model"]["architecture"],
        config["model"]["num_classes"],
    ).to(device)

    train_loader, val_loader = get_dataloaders(
        data_dir=config["data"]["data_dir"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"].get("num_workers", 2),
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["training"]["learning_rate"]
    )
    criterion = nn.CrossEntropyLoss()

    checkpoint_dir = Path(config["output"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_path = checkpoint_dir / config["output"]["model_name"]

    best_val_loss = float("inf")
    patience_counter = 0
    patience = config["training"]["early_stopping_patience"]

    for epoch in range(config["training"]["epochs"]):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(json.dumps({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
        }), flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "architecture": config["model"]["architecture"],
                "num_classes": config["model"]["num_classes"],
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }, save_path)
            print(json.dumps({
                "event": "checkpoint_saved",
                "path": str(save_path),
            }), flush=True)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(json.dumps({
                    "event": "early_stopping",
                    "epoch": epoch + 1,
                }), flush=True)
                break

    print(json.dumps({
        "event": "training_complete",
        "best_val_loss": round(best_val_loss, 4),
    }), flush=True)


if __name__ == "__main__":
    main()
