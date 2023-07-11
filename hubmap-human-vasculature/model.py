'''
This is my version of SegNet, made in PyTorch
The original paper for SegNet can be found online, and it details the architecture behind this model
'''
import torch
import torch.nn as nn

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
    def _get_max_pooling_indices(unpooled: torch.Tensor, pooled: torch.Tensor) -> torch.Tensor:
        '''
        One of the special features of SegNet is that it uses skip connections to help with avoiding vanishing gradients.
        However, to save memory, these skip connections do not copy the entire feature map but rather the indices of each value that was pooled with MaxPool.
        This method retrives those indices to be used later on in the network.
        '''
        pass
    
    def forward(self, x):
        x = self.convs(x)
        pooled = self.pool(x)
        indices = self._get_max_pooling_indices(x, pooled)
        return pooled, indices
