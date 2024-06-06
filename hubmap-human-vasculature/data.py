import os
import torch
from PIL import Image
import numpy as np
import cv2
import time

class DeepGlobeRoadExtractionDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, transforms=None, target_transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms
        self.target_transforms = target_transforms

        self.img_paths = []
        self.mask_paths = []

        for i in range(int(len(os.listdir(self.img_dir)) / 2)):
            self.img_paths.append(os.path.join(self.img_dir, f'{int(i) + 1}_sat.png'))
            self.mask_paths.append(os.path.join(self.img_dir, f'{int(i) + 1}_mask.png'))

    def __len__(self):
        return len([file for file in os.listdir(self.img_dir) if os.path.isfile(os.path.join(self.img_dir, file))])

    def __getitem__(self, idx):
        image_paths = self.img_paths[idx]
        mask_paths = self.mask_paths[idx]
        images = []
        masks = []

        if type(idx) == int:
            image = Image.open(self.img_paths[idx])
            mask = Image.open(self.mask_paths[idx])
            images.append(image)
            masks.append(mask)
        else:
            for img_path, mask_path in zip(image_paths, mask_paths):
                image = Image.open(img_path)
                image = np.array(image)
                images.append(image)

                mask = Image.open(mask_path)
                mask = np.array(mask)
                masks.append(mask)

        # Using min-max normalization for feature scaling
        for image, mask in zip(images, masks):
            image = torch.from_numpy(np.array(image, dtype=np.float32))
            mask = torch.from_numpy(np.array(mask, dtype=np.float32))
            
            image_min, _ = torch.min(image, dim=-1, keepdim=True)
            image_max, _ = torch.max(image, dim=-1, keepdim=True)
            image = (image - image_min) / (image_max - image_min)
            
            mask_min, _ = torch.min(mask, dim=-1, keepdim=True)
            mask_max, _ = torch.max(mask, dim=-1, keepdim=True)
            mask = (mask - mask_min) / (mask_max - mask_min)
            
        images = torch.from_numpy(np.array(images, dtype=np.float32))
        masks = torch.from_numpy(np.array(masks, dtype=np.float32))

        if self.transforms:
            images = self.transforms(images)
        if self.target_transforms:
            masks = self.target_transforms(masks)

        return images, masks

def visualize_data(image: torch.Tensor):
    if image.shape[0] == 1:
        image = torch.squeeze(image, dim=0)
        print(image.shape)
        cv2.imshow('Image', image.numpy())
        time.sleep(20)
    else:
        pass

def test():
    dataset = DeepGlobeRoadExtractionDataset('data/train')
    test = dataset[0]
    for image in test:
        print(image.shape)

    print(len(dataset))

if __name__ == '__main__':
    test()
    # print(len(os.listdir('./data/train')))
