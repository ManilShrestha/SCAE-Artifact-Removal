from torch.utils.data import Dataset
import os
from PIL import Image
import torch
import numpy as np

class SCAEDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        # Load artifact images
        artifact_dir = os.path.join(root_dir, "artifact")
        for img_name in os.listdir(artifact_dir):
            if img_name.endswith(".jpg"):  # Assuming images are in jpg format
                self.images.append(os.path.join(artifact_dir, img_name))
                self.labels.append(1)  # Artifact

        # Load non-artifact images
        non_artifact_dir = os.path.join(root_dir, "non-artifact")
        for img_name in os.listdir(non_artifact_dir):
            if img_name.endswith(".jpg"):  # Assuming images are in jpg format
                self.images.append(os.path.join(non_artifact_dir, img_name))
                self.labels.append(0)  # Non-artifact

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return torch.unsqueeze(torch.tensor(np.array(image)), dim=0), label