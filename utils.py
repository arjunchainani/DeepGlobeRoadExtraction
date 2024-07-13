import os
import glob
import re
import itertools
from PIL import Image
import torch
from data import DeepGlobeRoadExtractionDataset

def clean_dataset(dir):
    '''
    Cleans the dataset folder by renaming numbered images to be in numerical order for easier Pytorch dataset definition
    '''
    # for file_num, file in enumerate(os.listdir('./data/train')):
    #     print(f'{file_num} {file}')

    for file in os.listdir(dir):
        if file[-4:] == ".jpg":
            image = Image.open(os.path.join(dir, file))
            image.save(os.path.join(dir, file[:-4] + ".png"), "PNG")
            os.remove(os.path.join(dir, file))
            del image
    
    files = glob.glob1(dir, "*.png")
    files = sorted(files, key=lambda x:float(re.findall("(\d+)",x)[0]))
    print(files)
    
    for file_num, file in list(zip(itertools.count(1, 0.5), files)):
        path = os.path.join(dir, file)
        os.rename(path, os.path.join(dir, str(int(file_num)) + file[file.index('_'):]))
    
    print(os.listdir(dir))

def get_dataloaders(
        train_dir,
        valid_dir, 
        transforms, 
        target_transforms=None,
        batch_size=32,
        pin_memory=False,
        num_workers=4
    ):
    '''
    Returns the train and validation dataloaders
    '''
    train_dataset = DeepGlobeRoadExtractionDataset(img_dir=train_dir, transforms=transforms, target_transforms=target_transforms)
    train_dl = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=None,
    )

    validation_dataset = DeepGlobeRoadExtractionDataset(img_dir=valid_dir)
    validation_dl = torch.utils.data.DataLoader(
        dataset=validation_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=None,
    )

    return train_dl, validation_dl

if __name__ == "__main__":
    pass
    # clean_dataset("./data/train")
