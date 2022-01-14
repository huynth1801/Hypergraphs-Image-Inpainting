from .base_model import BaseModel
from .gc_layer import GatedConv2d, GatedDeConv2d
import torch
import torch.nn as nn
from torch_geometric.nn.conv import HypergraphConv

class Generator(BaseModel):
    def __init__(self, args):
        super(Generator, self).__init__()
        channel = 64
        input_dim = 3
        self.init_weights()

        # Coarse network
        self.gated_conv1 = GatedConv2d(in_channels=input_dim+1, 
                                out_channels=channel, kernel_size=7, padding='same', dilation=1)
        self.elu1 = nn.ELU()

        # Encoder for coarse network
        skip_connections = []
        downsamples = 3
        for i in range(1, downsamples):
            x = GatedConv2d()
