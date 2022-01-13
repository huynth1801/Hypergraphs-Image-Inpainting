# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn.parameter import Parameter

# class HypergraphConv(nn.Module):
#     def __init__(self, 
#                 in_channels,
#                 out_channels,
#                 features_height,
#                 features_width,
#                 edges,
#                 filters=64,                                                                                               # Intermeditate channels for phi and lambda matrices - A Hyperparameter
#                 apply_bias=True,                                                                                          
#                 training=True, 
#                 name=None, 
#                 dtype=None, 
#                 dynamic=False,
#                 **kwargs):
#         super(HypergraphConv, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.features_height = features_height
#         self.features_width = features_width
#         self.vertices = self.features_height * self.features_width
#         self.edges = edges
#         self.apply_bias = apply_bias
#         self.training = training
#         self.filters = filters
#         self.initialize_weights()

#         self.phi_conv = nn.Conv2d(in_channels, self.filters, kernel_size=1, padding='same')
#         self.A_conv = nn.Conv2d(self.filters, self.filters, kernel_size=1, padding='same')
#         self.M_conv = nn.Conv2d(self.filters, self.edges, kernel_size=7, padding='same')

#         # Make a weight of size (input channels * output channels) for applying the hypergraph convolution
#         self.weight = nn.parameter.Parameter(
#                         nn.init.xavier_uniform_(torch.empty(self.in_channels * self.out_channels, dtype=torch.float32))
#         )

#          # If applying bias on the output features, make a weight of size (output channels) 
#         if self.apply_bias:
#              self.bias = nn.parameter.Parameter(
#                         nn.init.xavier_uniform_(torch.empty(self.out_channels, dtype=torch.float32))
#              )
    
#     def initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)


#     def forward(self, x):
#         # Phi matrix
#         phi = self.phi_conv(x)
#         phi = phi.view(self.filters, self.vertices, -1)

#         # Lambda matrix
