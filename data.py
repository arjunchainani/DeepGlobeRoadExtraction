import os
import torch
from PIL import Image, ImageOps
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
            image = Image.open(self.img_paths[idx]).convert('RGB')
            mask = Image.open(self.mask_paths[idx])
            mask = ImageOps.grayscale(mask)

            images.append(image)
            masks.append(mask)
        else:
            for img_path, mask_path in zip(image_paths, mask_paths):
                image = Image.open(img_path)
                image = np.array(image)
                images.append(image)

                mask = Image.open(mask_path)
                mask = ImageOps.grayscale(mask)
                mask = np.array(mask)
                masks.append(mask)

        # Using min-max normalization for feature scaling
        for index, (image, mask) in enumerate(zip(images, masks)):
            image = torch.from_numpy(np.array(image, dtype=np.float32))
            mask = torch.from_numpy(np.array(mask, dtype=np.float32))
            
            image_min, _ = torch.min(image, dim=-1, keepdim=True)
            image_max, _ = torch.max(image, dim=-1, keepdim=True)
            images[index] = (image - image_min) / (image_max - image_min)
            
            mask[mask == 255.0] = 1.0
            masks[index] = mask            
            
        images = torch.from_numpy(np.array(images, dtype=np.float32))
        masks = torch.from_numpy(np.array(masks, dtype=np.float32))

        images = torch.stack([image.reshape((image.shape[-1], image.shape[-3], image.shape[-2])) for image in images], dim=0)
        masks = torch.stack([mask.reshape((mask.shape[-1], mask.shape[-3], mask.shape[-2])) for mask in torch.unsqueeze(masks, dim=-1)], dim=0)

        if self.transforms:
            images = self.transforms(images)
            masks = self.transforms(masks)

        images = torch.squeeze(images) if images.shape[0] == 1 else images
        masks = torch.squeeze(masks) if masks.shape[0] == 1 else masks

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
