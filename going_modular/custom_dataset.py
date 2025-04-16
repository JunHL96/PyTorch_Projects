import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    """
    Custom dataset for loading images and labels from a CSV file.

    Args:
        csv_file (str): Path to CSV file containing image paths and labels.
        base_dir (str): Base directory where images are stored.
        transform (callable, optional): Optional transform to be applied on an image.
    """
    def __init__(self, csv_file, base_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.base_dir = base_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # 'file_name' in CSV contains the path relative to base_dir (e.g., "train_data/...")
        img_path = os.path.join(self.base_dir, row["file_name"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = int(row["label"])
        return image, label
