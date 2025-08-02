"""Dataset utilities."""

from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Image as HFImage

from utils.augment import get_transforms

class HFDatasetWrapper(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __getitem__(self, idx):
        example = self.hf_dataset[idx]
        img = example["image"]
        if self.transform:
            img = self.transform(img)
        label = example["label"]
        return img, label, idx

    def __len__(self):
        return len(self.hf_dataset)

def build_dataset(config):
    """Return base dataset and train/validation DataLoaders."""
    dataset_name = config.get("dataset")
    if dataset_name == "hecto25":
        dataset_name = "growingduck/hct25"

    dataset = load_dataset(dataset_name, split="train")
    dataset = dataset.cast_column("image", HFImage())
    split = dataset.train_test_split(test_size=config.get("val_split", 0.2), seed=42)

    transform_train, transform_eval = get_transforms(config)

    train_loader = DataLoader(
        HFDatasetWrapper(split["train"], transform_train),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        HFDatasetWrapper(split["test"], transform_eval),
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
    )

    return dataset, train_loader, val_loader