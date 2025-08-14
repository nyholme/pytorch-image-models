import torch
import numpy as np

from ._registry import register_model

from escnn import gspaces
from escnn import nn


class CNSteerableCNN(torch.nn.Module):
    def __init__(
        self,
        n_rot,
        n_in_channels=3,
        input_size=224,
        n_rep_channels=[10, 24],
        kernel_sizes=[5, 5],
        dense_hidden_dims=[64],
        n_classes=10
    ):
        super(CNSteerableCNN, self).__init__()
        assert len(n_rep_channels) == len(kernel_sizes) # the two lists should be of equal length, i.e. one less that the number of layers

        self.num_classes = n_classes
        
        self.r2_act = gspaces.rot2dOnR2(N=n_rot)

        self.feat_types = [nn.FieldType(self.r2_act,  n_in_channels*[self.r2_act.trivial_repr])]
        for n_reps in n_rep_channels:
            self.feat_types.append(nn.FieldType(self.r2_act,  n_reps*[self.r2_act.regular_repr]))

        self.conv_blocks = torch.nn.ModuleList([])
        for i in range(len(n_rep_channels)):
            block = self.get_conv_block(self.feat_types[i], self.feat_types[i + 1], kernel_sizes[i])
            self.conv_blocks.append(block)

        self.spatial_pool = nn.PointwiseAvgPoolAntialiased(self.feat_types[-1], sigma=0.66, stride=1)
        self.group_pool = nn.GroupPooling(self.feat_types[-1])

        dense_input_dim = n_rep_channels[-1] * (input_size - np.sum(np.array(kernel_sizes) - np.array(len(kernel_sizes) * [3])))**2

        dense_dims = [dense_input_dim] + dense_hidden_dims + [n_classes]

        self.dense_net = torch.nn.Sequential(torch.nn.Linear(dense_input_dim, dense_hidden_dims[0]))
        
        for i in range(1, len(dense_dims) - 1):
            self.dense_net.append(torch.nn.BatchNorm1d(dense_dims[i]))
            self.dense_net.append(torch.nn.ELU(inplace=True))
            self.dense_net.append(torch.nn.Linear(dense_dims[i], dense_dims[i + 1]))
            

    def get_conv_block(self, in_type, out_type, kernel_size):
        block = nn.SequentialModule(
            #nn.MaskModule(in_type, 29, margin=1),
            nn.R2Conv(in_type, out_type, kernel_size=kernel_size, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        return block


    def forward(self, x):
        x = nn.GeometricTensor(x, self.feat_types[0])

        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        # pool over the spatial dimensions
        x = self.spatial_pool(x)
        
        # pool over the group
        x = self.group_pool(x)

        x = x.tensor
        
        # classify with the final fully connected layers)
        x = self.dense_net(x.reshape(x.shape[0], -1))
        return x

@register_model
def cnsteerablecnn(pretrained: bool = False, **kwargs) -> CNSteerableCNN:
    """Constructs a C_N Steerable CNN model using layers from the ESCNN package.
    """
    if pretrained:
        raise ValueError("Support for pretrained ESCNN models not yet implemented.")
    print("Creating a C_N Steerable CNN model using default class constructor. If " \
          "something seem wrong, should probably implement using full " \
            "`build_model_with_cfg` wrapper.")

    default_model_args = dict(
        n_rot=1,
        n_in_channels=3,
        input_size=224,
        n_rep_channels=[10, 24, 24, 10],
        kernel_sizes=[5, 9, 9, 9],
        dense_hidden_dims=[64],
        n_classes=10
    )
    for key in ['pretrained_cfg', 'pretrained_cfg_overlay', 'cache_dir', 'in_chans', 'drop_rate']:
        kwargs.pop(key)
    model = CNSteerableCNN(**dict(default_model_args, **kwargs))
    return model
