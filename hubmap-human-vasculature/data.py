import os
import torch
import torchvision
from PIL import Image

class DeepGlobeRoadExtractionDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, transforms=None, target_transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms
        self.target_transforms = target_transforms

    def __len__(self):
        return len([file for file in os.listdir(self.img_dir) if os.path.isfile(os.path.join(self.img_dir, file))])

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, str(idx + 1) + "_sat.png")
        mask_path = os.path.join(self.img_dir, str(idx + 1) + "_mask.png")
        image = Image.open(img_path)
        mask = Image.open(mask_path)

        if self.transforms:
            image = self.transforms(image)
        if self.target_transforms:
            mask = self.target_transforms(mask)

        return image, mask
    
def test():
    dataset = DeepGlobeRoadExtractionDataset('data/train')
    print(len(dataset))
    print(dataset[0])
    # print(dataset[0:5].shape)

if __name__ == '__main__':
    # test()

    torchvision.transforms.v2()