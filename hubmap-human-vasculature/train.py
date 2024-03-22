import torch
import torch.nn as nn
from model import SegNet

class HyperParameters():
    def __init__(self) -> None:
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.TRAIN_DIR = 'data/train'
        self.TEST_DIR = 'data/test'
        