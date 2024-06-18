'''
@author: Abder-Rahman Ali, PhD
@email: aali25@mgh.harvard.edu
@date: 2024
'''

import os
import glob
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import pandas as pd

class CosineKMeans(KMeans):
    def _transform(self, X):
        return pairwise_distances(X, self.cluster_centers_, metric='cosine')

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = glob.glob(self.root_dir + '/*.png')

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx])
        if self.transform:
            image = self.transform(image)
        return image, self.image_files[idx]

transform = transforms.Compose([transforms.Resize((512, 512)), 
                                transforms.Grayscale(num_output_channels=3),
                                transforms.ToTensor(), 
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

print("Loading images...")
dataset = ImageDataset(root_dir='./data/unlabeled/', transform=transform)

dataloader = DataLoader(dataset, batch_size=len(dataset))

images, image_names = next(iter(dataloader))

images = images.view(images.size(0), -1)

features = images.numpy()

print("Clustering images...")

kmeans = CosineKMeans(n_clusters=5, random_state=42)
kmeans.fit(features)

labels = kmeans.labels_

centroids = [np.argmin(pairwise_distances(features, center.reshape(1, -1), metric='cosine')) for center in kmeans.cluster_centers_]

distances = kmeans.transform(features)

centroid_distances = distances[np.arange(len(distances)), kmeans.labels_]

clusters = {i: [] for i in range(5)}
centroid_flags = {i: [] for i in range(5)}
outlier_flags = {i: [] for i in range(5)}
for i, label in enumerate(labels):
    clusters[label].append(os.path.basename(image_names[i]))
    centroid_flags[label].append(i in centroids)
    outlier_flags[label].append(centroid_distances[i])

for label in outlier_flags:
    outlier_flags[label] = np.argsort(outlier_flags[label])[-5:]

print("Saving images to CSV files...")

for label, image_names in clusters.items():
    df = pd.DataFrame({
        'image_name': image_names,
        'is_centroid': centroid_flags[label],
        'is_outlier': [i in outlier_flags[label] for i in range(len(image_names))]
    })
    df.to_csv(f'class_{label + 1}.csv', index=False)

print("All images saved to CSV files.")
