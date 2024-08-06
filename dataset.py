from torch.utils.data import Dataset, WeightedRandomSampler, Subset
from PIL import Image
import numpy as np
import os
import pandas as pd
import torch

# Define the custom dataset
class ISICDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.classes = sorted(set(labels))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_random_subset_without_given_indices(dataset, num_samples, indices_to_exclude):
    all_indices = list(range(len(dataset)))
    indices_to_include = list(set(all_indices) - set(indices_to_exclude))
    random_indices = np.random.choice(indices_to_include, num_samples, replace=False)

    random_subset = Subset(dataset, random_indices)

    return random_subset, random_indices

def load_isic_subset(total_train_size, augments, norm_only):
    metadata = pd.read_csv('data/metadata.csv')
    labels = metadata['malignant'].values.astype(int)
    files = [f"data/ISIC_2024_Training_Input/{f}" for f in os.listdir('data/ISIC_2024_Training_Input') if f.endswith('.jpg')]

    dataset_train = ISICDataset(files, labels, transform=augments)
    dataset_knn = ISICDataset(files, labels, transform=norm_only)

    targets = torch.tensor(labels)
    # get all indices where targets is 1
    positive_indices = torch.where(targets == 1)[0]

    # get 10.000 random indices where targets is 0
    negative_indices = torch.where(targets == 0)[0]
    negative_indices_train = negative_indices[torch.randperm(negative_indices.size(0))[:total_train_size-len(positive_indices)]]

    # combine positive and negative indices
    indices = torch.cat([positive_indices, negative_indices_train])

    train_dataset = Subset(dataset_train, indices)

    # get 50% of the positive indices
    positive_indices_knn_val = positive_indices[torch.randperm(positive_indices.size(0))[:len(positive_indices)//2]]

    # fill up to 1000 indices with negative indices
    negative_indices_knn_val = negative_indices[torch.randperm(negative_indices.size(0))[:1000-len(positive_indices_knn_val)]]

    knn_val_dataset = Subset(dataset_knn, torch.cat([positive_indices_knn_val, negative_indices_knn_val]))

    # get the rest of the positive indices
    positive_indices_knn_train = positive_indices[torch.randperm(positive_indices.size(0))[len(positive_indices)//2:]]

    # fill up to 5000 indices with negative indices
    negative_indices_knn_train = negative_indices[torch.randperm(negative_indices.size(0))[1000-len(positive_indices_knn_val):(5000-len(positive_indices_knn_train))+(1000-len(positive_indices_knn_val))]]

    knn_train_dataset = Subset(dataset_knn, torch.cat([positive_indices_knn_train, negative_indices_knn_train]))

    return train_dataset, knn_train_dataset, knn_val_dataset