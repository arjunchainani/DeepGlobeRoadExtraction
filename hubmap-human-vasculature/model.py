'''
This is my version of SegNet, made in PyTorch
The original paper for SegNet can be found online, and it details the architecture behind this model
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionLayers(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_layers: int) -> None:
        super(ConvolutionLayers, self).__init__()

        self.convs = nn.ModuleList()

        # Initial convolution layer - maps in_features to out_features
        self.convs.append(
            nn.Sequential(
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_features),
                nn.ReLU(inplace=True)
            )
        )

        for i in range(num_layers - 1):
            # The later convolution layers in each specific block keep the same number of features but just densify the image
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(out_features, out_features, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(out_features), 
                    nn.ReLU(inplace=True)
                )
            )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    @staticmethod
    def _max_pool_with_indices(image: torch.Tensor) -> torch.Tensor:
        '''
        One of the special features of SegNet is that it uses skip connections to help with avoiding vanishing gradients.
        However, to save memory, these skip connections do not copy the entire feature map but rather the indices of each value that was pooled with MaxPool.
        This method retrives those indices to be used later on in the network.
        '''
        return F.max_pool2d_with_indices(
            image,
            kernel_size=2,
            stride=2,
            padding=None,
            return_indices=True
        )
    
    def forward(self, x):
        x = self.convs(x)
        pooled, indices = self._max_pool_with_indices(x)
        return pooled, indices
    
class SegNet(nn.Module):
    def __init__(
            self, 
            in_features: int, 
            out_features: int, 
            channel_arch=[64, 128, 256, 512],
            num_filters=[2, 2, 3, 3, 3],
        ):
        '''
        This object contains the full architecture for the SegNet model
        
        Params:
        - in_features = How many channels are in the input image
        - out_features = How many channels to have in the output image
        - channel_arch = How many channels the image has as it passes through each of the network convolution layers
        - num_filters = How many convolutional filters to have in each VGG-16 style layer.
        '''
        super(SegNet, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.channel_arch = channel_arch
        self.num_filters = num_filters

        self.down_sampling = nn.ModuleList()
        self.up_sampling = nn.ModuleList()

        self.pooling_indices = []

        # Downsampling part of SegNet
        for (layer, channels), filters in zip(enumerate(self.channel_arch), self.num_filters):
            if layer == 0:
                self.down_sampling.append(
                    ConvolutionLayers(self.in_features, channels, num_layers=filters)
                )
            else:
                self.down_sampling.append(
                    ConvolutionLayers(self.channel_arch[layer - 1], channels, num_layers=filters)
                )

def test_model():
    model = SegNet(in_features=3, out_features=1)
    print(model)