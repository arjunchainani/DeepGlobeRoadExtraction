import os
import torch
import torchvision

class DeepGlobeRoadExtractionDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms

    def __len__(self):
        pass

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, idx, "_sat.jpg")
        mask_path = os.path.join(self.img_dir, idx, "_mask.png")
        image = torchvision.io.read_image(img_path)
        mask = torchvision.io.read_image(mask_path)