import torch
import torch.nn as nn
from model import SegNet

class HyperParameters():
    def __init__(self) -> None:
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.TRAIN_DIR = 'data/train'
        self.TEST_DIR = 'data/test'
        self.BATCH_SIZE = 44
        self.LEARNING_RATE = 1e-3
        self.ADJUSTED_IMAGE_WIDTH = 512
        self.ADJUSTED_IMAGE_HEIGHT = 512
        self.LOAD_MODEL = False
        self.PIN_MEMORY = True
        self.NUM_WORKERS = 4
    
def train(dl, model, optimizer, loss, scaler):
    pass # to be implemented soon