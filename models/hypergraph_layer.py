import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class HypergraphConv(nn.Module):
    def __init__(self, 
                in_features,
                out_features,
                features_height,
                features_width,
                edges,
                filters=64,                                                                                               # Intermeditate channels for phi and lambda matrices - A Hyperparameter
                apply_bias=True,                                                                                          
                training=True, 
                name=None, 
                dtype=None, 
                dynamic=False,
                **kwargs):
        super(HypergraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.features_height = features_height
        self.features_width = features_width
        self.vertices = self.features_height * self.features_width
        self.edges = edges
        self.apply_bias = apply_bias
        self.training = training
        self.filters = filters
        self.initialize_weights()

        self.phi_conv = nn.LazyConv2d(self.filters, kernel_size=1, padding='same')
        self.A_conv = nn.LazyConv2d(self.filters, kernel_size=1, padding='same')
        self.M_conv = nn.LazyConv2d(self.edges, kernel_size=7, padding='same')

        # Make a weight of size (input channels * output channels) for applying the hypergraph convolution
        self.weight = nn.parameter.Parameter(
                        nn.init.xavier_uniform_(torch.empty(self.in_channels * self.out_channels, dtype=torch.float32))
        )

         # If applying bias on the output features, make a weight of size (output channels) 
        if self.apply_bias:
             self.bias = nn.parameter.Parameter(
                        nn.init.xavier_uniform_(torch.empty(self.out_channels, dtype=torch.float32))
             )
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, x):
        # Phi matrix
        phi = self.phi_conv(x)
        phi = phi.view(-1, self.filters, self.vertices)

        # Lambda matrix
        A = F.adaptive_avg_pool2d(x, (1, 1))
        A = self.A_conv(A)
        A = torch.linalg(A)
        A = torch.diag(A.squeeze())

        # Omega matrix
        M = self.M_conv(x)
        M = M.view(-1, self.vertices, self.edges)

        