from gc_layer import GatedConv2d
import torch
import torch.nn as nn
from base_model import BaseModel


#-----------------------------------------------
#                  Discriminator
#-----------------------------------------------
# Input: generated image / ground truth and mask
class Discriminator(BaseModel):
    def __init__(self, channels):
        super(Discriminator, self).__init__()
        self.channels = channels
        self.init_weights()

        self.disc = nn.Sequential(GatedConv2d(4, channels, kernel_size=3, dilation=1, pad_type='zero', padding=1, activation='LeakyReLU'),
        GatedConv2d(channels, channels*2, kernel_size=3, dilation=1, pad_type='zero',padding=1, activation='LeakyReLU'),
        GatedConv2d(channels*2, channels*4, kernel_size=3, dilation=1, pad_type='zero', padding=1, activation='LeakyReLU'),
        GatedConv2d(channels*4, channels*8, kernel_size=3, dilation=1, pad_type='zero', padding=1, activation='LeakyReLU'),
        GatedConv2d(channels*8, channels*8, kernel_size=3, dilation=1, pad_type='zero', padding=1, activation='LeakyReLU'),
        GatedConv2d(channels*8, channels*8, kernel_size=3, dilation=1, pad_type='zero', padding=1, activation='LeakyReLU'),
        GatedConv2d(channels*8, 1, kernel_size=3, dilation=1, pad_type='zero', padding=1, activation='LeakyReLU'),

        )
    def forward(self, inputs, mask):
         # the input x should contain 4 channels because it is a combination of recon image and mask
        x = torch.cat((inputs, mask), dim=1)    # In: [B, 4, H, W]
        output = self.disc(x)
        return output


if __name__ == "__main__":
    model = Discriminator(64)
    print(model)
    from torchsummary import summary
    print(summary(model, [(3,256,256), (1,256,256)]))