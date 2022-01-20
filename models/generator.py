from base_model import BaseModel
from gc_layer import GatedConv2d, TransposeGatedConv2d
import torch
import torch.nn as nn
from torch_geometric.nn.conv import HypergraphConv


#-----------------------------------------------
#                   Generator
#-----------------------------------------------
# Input: masked image + mask
# Output: filled image
class Generator(BaseModel):
    def __init__(self, channels):
        super(Generator, self).__init__()
        self.init_weights()

        #### Coarse network #####
        self.gated_conv1 = GatedConv2d(in_channels=4, 
                                out_channels=channels, kernel_size=7, stride=1, padding=3, dilation=1, pad_type='zero', activation='ELU')

        # Encoder for coarse network
        skip_connections = []
        self.gated_conv2 = GatedConv2d(in_channels=64, 
                                out_channels=128, kernel_size=3, stride=2, padding=1, dilation=1, pad_type='zero', activation='ELU')

        self.gated_conv3 = GatedConv2d(in_channels=128, 
                                out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, pad_type='zero', activation='ELU')

        self.gated_conv4 = GatedConv2d(in_channels=128, 
                                out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, pad_type='zero', activation='ELU')

        self.gated_conv5 = GatedConv2d(in_channels=128, 
                                out_channels=256, kernel_size=3, stride=2, padding=1, dilation=1, pad_type='zero', activation='ELU')

        self.gated_conv6 = GatedConv2d(in_channels=256, 
                                out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, pad_type='zero', activation='ELU')

        self.gated_conv7 = GatedConv2d(in_channels=256, 
                                out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, pad_type='zero', activation='ELU')

        # Center Convolutions for higher receptive field
        # These convolutions are with dilation=2
        # Can kiem tra lai padding = 'same'
        self.gated_conv8 = GatedConv2d(in_channels=256, 
                                out_channels=256, kernel_size=3, stride=1, padding=2, dilation=2, pad_type='zero', activation='ELU')

        self.gated_conv9 = GatedConv2d(in_channels=256, 
                                out_channels=256, kernel_size=3, stride=1, padding=2, dilation=2, pad_type='zero', activation='ELU')

        self.gated_conv10 = GatedConv2d(in_channels=256, 
                                out_channels=256, kernel_size=3, stride=1, padding=2, dilation=2, pad_type='zero', activation='ELU')

        # Decoder Network for Coarse Network
        self.gated_conv11 = GatedConv2d(in_channels=256, 
                                out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, pad_type='zero', activation='ELU')

        self.gated_conv12 = GatedConv2d(in_channels=256, 
                                out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, pad_type='zero', activation='ELU')

        self.gated_conv13 = GatedConv2d(in_channels=256, 
                                out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, pad_type='zero', activation='ELU')

        self.gated_conv14 = GatedConv2d(in_channels=128, 
                                out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, pad_type='zero', activation='ELU')

        self.gated_conv15 = GatedConv2d(in_channels=128, 
                                out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, pad_type='zero', activation='ELU')

        self.gated_conv16 = GatedConv2d(in_channels=128, 
                                out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, pad_type='zero', activation='ELU')

        # Concat with skip connection
        ###
        self.gated_conv17 = GatedConv2d(in_channels=64, 
                                out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, pad_type='zero', activation='ELU')

        # Coarse out
        self.gated_conv18 = GatedConv2d(in_channels=64, 
                            out_channels=3, kernel_size=3, stride=1, padding=1, dilation=1, pad_type='zero', activation='ELU')


        ##### Refine network #####
        # Dau tien chung ta can concat 
        self.refine1 = GatedConv2d(in_channels=4,
                        out_channels=64, kernel_size=7, stride=1, dilation=1, padding=3, pad_type='zero', activation='ELU')
        
        # Encoder for refine network
        # H/2, W/2
        self.refine2 = GatedConv2d(in_channels=64, out_channels=128, 
                        kernel_size=3, stride=2, padding=1, dilation=1, pad_type='zero', activation='ELU')
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.refine3 = GatedConv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1,
                        padding=1, dilation=1, pad_type='zero')
        self.refine4 = GatedConv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1,
                        padding=1, dilation=1, pad_type='zero', activation='ELU')
        self.refine5 = GatedConv2d(in_channels=128, out_channels=256, kernel_size=3,
                            stride=2, padding=1, dilation=1, pad_type='zero', activation='ELU')
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.refine6 = GatedConv2d(in_channels=256, out_channels=256, kernel_size=3,
                            stride=1, padding=1, dilation=1, pad_type='zero', activation='ELU')
        self.refine7 = GatedConv2d(in_channels=256, out_channels=256, kernel_size=3,
                            stride=1, padding=1, dilation=1,pad_type='zero', activation='ELU')
        self.refine8 = GatedConv2d(in_channels=256, out_channels=512, kernel_size=3,
                            stride=2, padding=1, dilation=1, pad_type='zero', activation='ELU')
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.refine9 = GatedConv2d(in_channels=512, out_channels=512, kernel_size=3, 
                            stride=1, padding=2, dilation=2, pad_type='zero', activation='ELU')
        self.refine10 = GatedConv2d(in_channels=512, out_channels=512, kernel_size=3,
                            stride=1, padding=2, dilation=2, pad_type='zero', activation='ELU')
        self.refine11 = GatedConv2d(in_channels=512, out_channels=512, kernel_size=3,
                            stride=1, padding=2, dilation=2, pad_type='zero', activation='ELU')
        self.refine12 = GatedConv2d(in_channels=512, out_channels=1024, kernel_size=3,
                            stride=1, padding=1, dilation=1, pad_type='zero', activation='ELU')
        self.refine13 = GatedConv2d(in_channels=1024, out_channels=1024, kernel_size=3,
                            stride=1, padding=1, dilation=1, pad_type='zero', activation='ELU')
        self.refine14 = GatedConv2d(in_channels=1024, out_channels=1024, kernel_size=3,
                            stride=1, padding=1, dilation=1, pad_type='zero', activation='ELU')

        # Apply Hypergraph convolution on last skip connections
        self.graph1 = HypergraphConv(in_channels=256, out_channels=512, num_edges=128, training=True)
        self.elu_g1 = nn.ELU()
        self.graph2 = HypergraphConv(in_channels=128, out_channels=128, num_edges=128, training=True)
        self.elu_g2 = nn.ELU()



    def forward(self, inputs, mask):
        x = torch.cat([inputs, mask], dim=1)
        x = self.gated_conv1(x)
        x = self.model(x)
        x = self.skip_connections(x)
        return x



if __name__=='__main__':
    model = Generator(64)
    print(model)
