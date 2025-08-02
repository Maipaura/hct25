import sys
import os
import argparse
import yaml
import torch
import torch.nn.functional as F
import timm
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Image as HFImage

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.augment import get_transforms


class TestDataset(Dataset):
    """Dataset wrapper for inference without labels."""

    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        example = self.hf_dataset[idx]
        img = example["image"]
        if self.transform:
            img = self.transform(img)
        return img, example["ID"]


def main(config, weights_path, sample_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sample_submission = pd.read_csv(sample_path)
    class_names = [col for col in sample_submission.columns if col != "ID"]
    num_classes = len(class_names)

    test_dataset_name = config.get("test_dataset", "change this as path to test_dataset")
    test_dataset = load_dataset(test_dataset_name, split="train")
    test_dataset = test_dataset.cast_column("image", HFImage())

    _, transform_eval = get_transforms(config)

    test_loader = DataLoader(
        TestDataset(test_dataset, transform_eval),
        batch_size=config.get("batch_size", 32),
        shuffle=False,
        num_workers=4,
    )

    model = timm.create_model(
        config["model_name"], pretrained=False, num_classes=num_classes
    )
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    temperature = config.get("temperature", 1.0)

    all_probs, all_ids = [], []
    with torch.no_grad():
        for inputs, ids in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs) / temperature
            probs = F.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_ids.extend(ids)

    probs_concat = np.concatenate(all_probs, axis=0)
    submission_df = pd.DataFrame(probs_concat, columns=class_names)
    submission_df.insert(0, "ID", all_ids)
    submission_df.to_csv(output_path, index=False)
    print(f"✅ Submission file saved as {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument(
        "--config", default="configs/config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--weights", default="best_model_ema.pth", help="Path to model weights"
    )
    parser.add_argument(
        "--sample", default="sample_submission.csv", help="Sample submission file"
    )
    parser.add_argument(
        "--output", default="submission.csv", help="Output submission file"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    main(config, args.weights, args.sample, args.output)
