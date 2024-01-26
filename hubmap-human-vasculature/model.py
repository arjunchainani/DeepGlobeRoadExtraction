'''
This is my version of SegNet, made in PyTorch
The original paper for SegNet can be found online, and it details the architecture behind this model
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_conv_layers: int) -> None:
        super(ConvBlock, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_conv_layers = num_conv_layers

        self.convs = nn.ModuleList()

        # Initial convolution layer - maps in_features to out_features
        self.convs.append(
            nn.Sequential(
                nn.Conv2d(self.in_features, self.out_features, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.out_features),
                nn.ReLU(inplace=True)
            )
        )

        for _ in range(num_conv_layers - 1):
            # The later convolution layers in each specific block keep the same number of features but just densify the image
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(self.out_features, self.out_features, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(self.out_features), 
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
            padding=(0),
            return_indices=True
        )
    
    def forward(self, x):
        for module in self.convs:
            x = module(x)

        pooled, indices = self._max_pool_with_indices(x)
        return pooled, indices

class Upsample(ConvBlock):
    def __init__(self, in_features: int, out_features: int, num_conv_layers: int) -> None:
        super(Upsample, self).__init__(in_features, out_features, num_conv_layers)

        self.indices = []
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.convs.insert(index=0, module=self.unpool)

    # Overrides from ConvBlock class
    def forward(self, x, indices):
        self.indices = indices
        for module_num, module in enumerate(self.convs):
            if module_num == 0:
                x = module(x, self.indices)
            else:
                x = module(x)

        return x

    @staticmethod
    def _map_indices(image: torch.Tensor, new_img_size: tuple, index_map: torch.Tensor) -> torch.Tensor:
        '''
        Maps the pooled indices to a larger image using their old positions
        Any extra spots in the image will initially be filled with zeros and then densified later on through convolutions
        '''
        new_img = torch.zeros(new_img_size)

        # This section converts the indices returned by F.max_pool2d_with_indices into a format that we can use to map pixels in our image
        coors = []

        dim = new_img_size[0]
        index_map = torch.squeeze(index_map)
        index_map = torch.flatten(index_map).tolist()

        for index in index_map:
            x = index // dim
            y = index % dim
            coors.append((x, y))
        
        print(coors[0:5])
        print(f'Len: {len(coors)}')

    @staticmethod
    def _get_max_index(indices: torch.Tensor) -> int:
        '''
        Just an additional method for testing
        Returns the pooling index of any pixel in the pooled image
        '''
        indices = torch.flatten(indices).tolist()
        return max(indices)

class SegNet(nn.Module):
    def __init__(
            self, 
            in_features: int, 
            out_features: int, 
            channel_arch=[64, 128, 256, 512, 1024],
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

        self.pooling_indices = []  # Keeps track of the pooling indices from the max-pool layer in each convolution block

        # Downsampling part of SegNet
        for (layer, channels), filters in zip(enumerate(self.channel_arch), self.num_filters):
            if layer == 0:
                self.down_sampling.append(
                    ConvBlock(self.in_features, channels, num_conv_layers=filters)
                )
            else:
                self.down_sampling.append(
                    ConvBlock(self.channel_arch[layer - 1], channels, num_conv_layers=filters)
                )

        self.up_channel_arch = list(reversed(self.channel_arch))
        self.up_num_filters = list(reversed(self.num_filters))
        self.reverse_indices = list(reversed(self.pooling_indices))

        # Upsampling part of SegNet
        for (layer, channels), filters in zip(enumerate(self.up_channel_arch), self.up_num_filters):
            if layer == len(self.up_channel_arch) - 1: # Needs to be altered to be the last layer and use out_features instead
                self.up_sampling.append(
                    Upsample(channels, self.in_features, num_conv_layers=filters)
                )
            else:
                self.up_sampling.append(
                    Upsample(channels, self.up_channel_arch[layer + 1], num_conv_layers=filters)
                )

    def forward(self, x):        
        # Passing input through the downsampling layers
        for down in self.down_sampling:
            x, indices = down(x)
            self.pooling_indices.append(indices)

        self.reverse_indices = list(reversed(self.pooling_indices))

        for up, indices in zip(self.up_sampling, self.reverse_indices):
            # Now passing through upsampling layers
            x = up(x, indices)

        x = F.softmax(x, dim=0)

        return x

def test_model():
    input = torch.randn((2, 3, 512, 512))
    model = SegNet(in_features=3, out_features=1)
    result = model(input)
    print(result.shape)


if __name__ == '__main__':
    test_model()
