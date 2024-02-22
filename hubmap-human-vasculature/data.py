import os
import torch
import torchvision
from PIL import Image
import numpy as np

class DeepGlobeRoadExtractionDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, transforms=None, target_transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms
        self.target_transforms = target_transforms

        self.img_paths = []

        print(len(os.listdir(self.img_dir)) / 2)

        for i in range(len(os.listdir(self.img_dir))):
            j = i / 2
            if int(j) == j:
                self.img_paths.append(os.path.join(self.img_dir, f'{int(j) + 1}_mask.png'))
            else:
                self.img_paths.append(os.path.join(self.img_dir, f'{int(j) + 1}_sat.png'))

    def __len__(self):
        return len([file for file in os.listdir(self.img_dir) if os.path.isfile(os.path.join(self.img_dir, file))])

    def __getitem__(self, idx):
        print(f'-------- {self.img_paths[0]}')
        image_paths = self.img_paths[idx]
        images = []

        for path in image_paths:
            image = Image.open(path)
            image = np.array(image)
            images.append(image)

        images = np.array(images)
        images = torch.tensor(images)
        print(images.shape)
        #
        # if self.transforms:
        #     image = self.transforms(image)
        # if self.target_transforms:
        #     mask = self.target_transforms(mask)

        return images


def test():
    dataset = DeepGlobeRoadExtractionDataset('data/train')
    # print(len(dataset))
    print(dataset[0])
    # print(dataset[0:5])


if __name__ == '__main__':
    test()
    # print(len(os.listdir('./data/train')))
