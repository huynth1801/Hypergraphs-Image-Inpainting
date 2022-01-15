from gc_layer import GatedConv2d
import torch
import torch.nn as nn
from base_model import BaseModel

class Discriminator(BaseModel):
    def __init__(self, channels = 64):
        super(Discriminator, self).__init__()
        self.channels = channels
        input_dim = 3
        self.init_weights()

        self.gt_conv1 = GatedConv2d(input_dim+1, channels, kernel_size=3, dilation=1, padding='same')
        self.lk1 = nn.LeakyReLU()

        self.gt_conv2 = GatedConv2d(channels, channels*2, kernel_size=3, dilation=1, padding='same')
        self.lk2 = nn.LeakyReLU()

        self.gt_conv3 = GatedConv2d(channels*2, channels*4, kernel_size=3, dilation=1, padding='same')
        self.lk3 = nn.LeakyReLU()

        self.gt_conv4 = GatedConv2d(channels*4, channels*8, kernel_size=3, dilation=1, padding='same')
        self.lk4 = nn.LeakyReLU()

        self.gt_conv5 = GatedConv2d(channels*8, channels*8, kernel_size=3, dilation=1, padding='same')
        self.lk5 = nn.LeakyReLU()

        self.gt_conv6 = GatedConv2d(channels*8, channels*8, kernel_size=3, dilation=1, padding='same')
        self.lk6 = nn.LeakyReLU()

        self.gt_conv7 = GatedConv2d(channels*8, channels*8, kernel_size=3, dilation=1, padding='same')
        self.lk7 = nn.LeakyReLU()

    def forward(self, inputs, mask):
        x_in = torch.cat([inputs, mask], dim=1)
        output = self.gt_conv1(x_in)
        output = self.lk1(output)
        output = self.gt_conv2(output)
        output = self.lk2(output)
        output = self.gt_conv3(output)
        output = self.lk3(output)
        output = self.gt_conv4(output)
        output = self.lk4(output)
        output = self.gt_conv5(output)
        output = self.lk5(output)
        output = self.gt_conv6(output)
        output = self.lk6(output)
        output = self.gt_conv7(output)
        output = self.lk7(output)
        return output


if __name__ == "__main__":
    model = Discriminator()
    print(model.print_network())
    from torchsummary import summary
    print(summary(model, [(3,256,256), (1,256,256)], 1))