from re import A

from markupsafe import re
from sklearn.neighbors import RadiusNeighborsClassifier
from base_model import BaseModel
from gc_layer import GatedConv2d, TransposeGatedConv2d
import torch
import torch.nn as nn
import torch.nn.functional as F
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

        self.gated_deconv13 = TransposeGatedConv2d(in_channels=256, 
                                out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, pad_type='zero', activation='ELU', sn=False)
        # Concatenate with 4
        self.gated_conv14 = GatedConv2d(in_channels=256, 
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
        # Dau tien chung ta can concatenate
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
        self.refine12 = GatedConv2d(in_channels=512, out_channels=512, kernel_size=3,
                            stride=1, padding=1, dilation=1, pad_type='zero', activation='ELU')
        self.refine13 = GatedConv2d(in_channels=512, out_channels=512, kernel_size=3,
                            stride=1, padding=1, dilation=1, pad_type='zero', activation='ELU')
        self.refine14 = GatedConv2d(in_channels=512, out_channels=512, kernel_size=3,
                            stride=1, padding=1, dilation=1, pad_type='zero', activation='ELU')

        # Apply Hypergraph convolution on last skip connections
        self.graph1 = HypergraphConv(in_channels=16, out_channels=512)
        self.elu_g1 = nn.ELU()
        self.graph2 = HypergraphConv(in_channels=256, out_channels=128)
        self.elu_g2 = nn.ELU()

        # Doing the first Deconvolution operation
        self.de_gt1 = TransposeGatedConv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,
                                padding=1, dilation=1, pad_type='zero', activation='ELU', sn=False)
        #  Concaternate

        # Decoder for refine network
        self.dec_rf1 = GatedConv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,
                                    padding=1, dilation=2, pad_type='zero', activation='ELU')

        self.dec_rf2 = GatedConv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,
                                    padding=1, dilation=2, pad_type='zero', activation='ELU')
        self.dec_rf3 = GatedConv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,
                                    padding=1, dilation=2, pad_type='zero', activation='ELU')
        self.dec_rf4 = TransposeGatedConv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1,
                                    padding=1, dilation=1, pad_type='zero', activation='ELU', sn=False)
        
        # concat x4 with skip[1][0]
        self.dec_rf5 = GatedConv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1,
                                    padding=1, dilation=1, pad_type='zero', activation='ELU')
        self.dec_rf6 = GatedConv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1,
                                    padding=1, dilation=1, pad_type='zero', activation='ELU')
        self.dec_rf7 = TransposeGatedConv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1,
                                    padding=1, dilation=1, pad_type='zero', activation='ELU', sn=False)
        # Concat x7 with skip[0][0]
        self.dec_rf8 = GatedConv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1,
                                    padding=1, dilation=1, pad_type='zero', activation='ELU')
        self.dec_rf9 = GatedConv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1,
                                    padding=1, dilation=1, pad_type='zero', activation='ELU')
        self.dec_rf10 = TransposeGatedConv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1,
                                    padding=1, dilation=1, pad_type='zero', activation='ELU')
        
        self.dec_rf11 = GatedConv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                                    padding=1, dilation=1, pad_type='zero', activation='ELU')
        self.dec_rf12 = GatedConv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                                    padding=1, dilation=1, pad_type='zero', activation='ELU')
        self.rf_out = GatedConv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1,
                                    padding=1, dilation=1, pad_type='zero', activation='none')
    
    
    def forward(self, img, mask):
        # img: entire img
        # mask: 1 for mask region; 0 for unmask region
        # Coarse
        skip_connections = []
        first_masked_img = img * (1 - mask) + mask
        x = torch.cat([first_masked_img, mask], dim=1)
        x = self.gated_conv1(x)
        x = self.gated_conv2(x)
        x = self.gated_conv3(x)
        x = self.gated_conv4(x)
        skip_connections.append(x)
        x = self.gated_conv5(x)
        x = self.gated_conv6(x)
        x = self.gated_conv7(x)
        x = self.gated_conv8(x)
        x = self.gated_conv9(x)
        x = self.gated_conv10(x)
        # Decoder for coarse
        x = self.gated_conv11(x)
        x = self.gated_conv12(x)
        x = self.gated_deconv13(x)
        x = torch.cat((x, skip_connections[0]), dim=1)
        x = self.gated_conv14(x)        
        x = self.gated_conv15(x)
        x = self.gated_conv16(x)
        x = self.gated_conv17(x)
        coarse_out = self.gated_conv18(x)
        coarse_out = F.interpolate(coarse_out, (img.shape[2], img.shape[3]))
        # Refine network
        sencond_masked_img = img * (1 - mask) + coarse_out * mask
        second_in = torch.cat((sencond_masked_img, mask), dim=1)
        refine_conv = self.refine1(second_in)
        refine_conv = self.refine2(refine_conv)
        refine_conv = self.pool1(refine_conv)
        refine_conv = self.refine3(refine_conv)
        refine_conv = self.refine4(refine_conv)
        x4 = refine_conv
        skip_connections.append(x4)
        refine_conv = self.refine5(refine_conv)
        refine_conv = self.refine6(refine_conv)
        refine_conv = self.refine7(refine_conv)
        x7 = refine_conv
        skip_connections.append(x7)
        refine_conv = self.refine8(refine_conv)
        refine_conv = self.refine9(refine_conv)
        refine_conv = self.refine10(refine_conv)
        refine_conv = self.refine11(refine_conv)
        x11 = refine_conv
        skip_connections.append(x11)
        refine_conv = self.refine12(refine_conv)
        refine_conv = self.refine13(refine_conv)
        refine_conv = self.refine14(refine_conv)
        x11 = self.graph1(x11, hyperedge_index=refine_conv.squeeze().long())
        x11 = self.elu_g1(x11)
        print("DONE")
        x7 = self.graph2(x7)
        x7 = self.elu_g2(x7)

        # Doing the first Deconvolution operation
        refine_conv = self.de_gt1(refine_conv)
        refine_conv = torch.cat((refine_conv, x11), dim=1)
        refine_conv = self.dec_rf1(refine_conv)
        refine_conv = self.dec_rf2(refine_conv)
        refine_conv = self.dec_rf3(refine_conv)
        refine_conv = self.dec_rf4(refine_conv)
        refine_conv = torch.cat((x7, refine_conv), dim=1)
        refine_conv = self.dec_rf5(refine_conv)
        refine_conv = self.dec_rf6(refine_conv)
        refine_conv = self.dec_rf7(refine_conv)
        refine_conv = torch.cat((x4, refine_conv), dim=1)
        refine_conv = self.dec_rf8(refine_conv)
        refine_conv = self.dec_rf8(refine_conv)
        refine_conv = self.dec_rf9(refine_conv)
        refine_conv = self.dec_rf10(refine_conv)
        refine_conv = self.dec_rf11(refine_conv)
        refine_conv = self.dec_rf12(refine_conv)
        refine_out = self.rf_out(refine_conv)
        refine_out = F.interpolate(refine_out, (img.shape[2], img.shape[3]))
        return coarse_out, refine_out



if __name__=='__main__':
    model = Generator(64)
    # print(model)
    from torchsummary import summary
    print(summary(model, [(3,256,256), (1,256,256)]))
