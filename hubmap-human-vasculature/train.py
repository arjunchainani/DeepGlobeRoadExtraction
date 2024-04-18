import torch
import torch.nn as nn
from model import SegNet

from tqdm import tqdm 

class HyperParameters():
    def __init__(self) -> None:
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.TRAIN_DIR = 'data/train'
        self.VALID_DIR = 'data/valid'
        self.TEST_DIR = 'data/test'
        self.BATCH_SIZE = 44
        self.LEARNING_RATE = 1e-3
        self.ADJUSTED_IMAGE_WIDTH = 512
        self.ADJUSTED_IMAGE_HEIGHT = 512
        self.LOAD_MODEL = False
        self.PIN_MEMORY = True
        self.NUM_WORKERS = 4
    
def train(dl, model, optimizer, loss, scaler):
    training_loop = tqdm(dl)
    params = HyperParameters()

    for batch_num, (images, real_masks) in enumerate(training_loop):
        images = images.to(params.DEVICE)
        real_masks = real_masks.to(params.DEVICE)

        if params.DEVICE == 'cuda':
            with torch.cuda.amp.autocast():
                pred_masks = model(images)
                loss = loss(pred_masks, real_masks)
        else:
            pred_masks = model(images)
            loss = loss(pred_masks, real_masks)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        training_loop.set_postfix(loss=loss.item())

def main():
    params = HyperParameters()