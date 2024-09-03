import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import cv2
import torch
from sklearn.model_selection import train_test_split

class Driving_Dataset(Dataset):
    def __init__(self, dataset_path, transform, is_train, train_ratio = 0.8, random_seed = 42):
        data_df = pd.read_csv(dataset_path)
        X = data_df['image'].values
        y = data_df['steering'].values
        y = y.astype(np.float32)

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, random_state=random_seed)
        if is_train:
            self.image_paths = X_train
            self.target = y_train
        else:
            self.image_paths = X_test
            self.target = y_test
        self.transform = transform

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        steering = self.target[index]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

        if self.transform:
            image = self.transform(image)

        return image, steering

if __name__ == "__main__":
    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((66, 200), antialias=True),
        v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = Driving_Dataset(dataset_path='dataset/augmented_driving_log.csv', transform=transform, is_train=True)

    image, steering = dataset[10]
    print(steering)
    # print(image.shape)
    # print(steering.shape)
