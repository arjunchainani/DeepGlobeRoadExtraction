import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from model import SegNet

from tqdm import tqdm 
from torchvision.transforms import v2
from torchmetrics.classification import Dice
import utils

# Stores all the hyperparameters so that they can be easily altered during the training process
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
        self.NUM_EPOCHS = 3
    
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

if __name__ == '__main__':
    params = HyperParameters()

    # Needs to be fixed
    transforms = v2.Compose([
        v2.Resize((params.ADJUSTED_IMAGE_HEIGHT, params.ADJUSTED_IMAGE_WIDTH)),
        v2.RandomHorizontalFlip(p=1),
        v2.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            inplace=False,
        ),
        v2.ToDtype(torch.float32, scale=True),
    ])

    model = SegNet(in_features=3, out_features=1).to(params.DEVICE)
    loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.LEARNING_RATE)

    train_dl, validation_dl = utils.get_dataloaders(
        params.TRAIN_DIR,
        params.VALID_DIR,
        transforms,
        batch_size=params.BATCH_SIZE,
        pin_memory=params.PIN_MEMORY,
        num_workers=params.NUM_WORKERS,
    )

    if params.LOAD_MODEL:
        model.load_state_dict(torch.load('./checkpoints/current.pth.tar'))

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(params.NUM_EPOCHS):
        train(train_dl, model, optimizer, loss, scaler)

        torch.save(model, './checkpoints/current.pth.tar')

        # Evaluating the dice score of the model 
        dice = Dice(average='macro')
        
        model.eval()
        with torch.no_grad():
            for x, y in train_dl:
                x = x.to(params.DEVICE)
                y = y.to(params.DEVICE)
                y_hat = model(x)

                dice_score = dice(y_hat, y)

        # Tensorboard Summary Writer
        writer = SummaryWriter(comment=f'LR_{params.LEARNING_RATE}_BATCHSIZE_{params.BATCH_SIZE}')
        writer.add_scalar("Loss: ", loss, global_step=epoch)
        writer.add_scalar("Dice Score: ", dice_score, global_step=epoch)

    writer.flush()
    writer.close()