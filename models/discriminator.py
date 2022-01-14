from gc_layer import GatedConv2d, GatedDeConv2d
import torch
import torch.nn as nn
from base_model import BaseModel

class Discriminator(BaseModel):
    def __init__(self, channels = 64, **kwargs):
        super(Discriminator, self).__init__()
        self.channels = channels
        input_dim = 3
        self.init_weights()

        self.gt_conv1 = GatedConv2d(input_dim+1, channels, kernel_size=3, dilation=1, padding='same')
        self.lk1 = nn.LeakyReLU()

        layers = []
        for i in range(1,7):
            mult = (2**i) if (2**i) < 8 else 8
            in_mult = (2**i) if (2**i) < 9 else 16
            layers.append(nn.Sequential(GatedConv2d(channels*in_mult, channels*mult, kernel_size=3, stride=2, padding='same', dilation=1),
                            nn.LeakyReLU()))

        self.layers = nn.Sequential(*layers)

    def forward(self, inputs, **kwargs):
        x = inputs[0]
        mask = inputs[1]
        x_in = torch.cat([x, mask], dim=1)
        output = self.gt_conv1(x_in)
        output = self.lk1(output)
        for i in range(1,7):
            output = self.layers[i](output)
        return output


if __name__ == "__main__":
    model = Discriminator()
    print(model)