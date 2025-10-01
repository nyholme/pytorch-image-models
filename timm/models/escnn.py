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
          "something seems wrong, should probably implement using full " \
            "`build_model_with_cfg` wrapper.")

    default_model_args = dict(
        n_rot=1,
        n_in_channels=3,
        input_size=128,
        n_rep_channels=[10, 24, 24, 10],
        kernel_sizes=[5, 9, 9, 9],
        dense_hidden_dims=[64],
        n_classes=10
    )
    for key in ['pretrained_cfg', 'pretrained_cfg_overlay', 'cache_dir', 'in_chans', 'drop_rate']:
        kwargs.pop(key)
    model = CNSteerableCNN(**dict(default_model_args, **kwargs))
    return model



# This is our draft scalable Steerable CNN model using a downsampler, fixed-res layers, and a readout.
"""
Steerable CNN model using the `escnn` package.

Model constructor arguments:
- L: input image size (int) (expects square images L x L)
- group_name: one of 'C_1', 'C_4', 'D_1', 'D_4'
- N: number of intermediate fixed-resolution steerable layers (int)

Design:
- Input: L x L ImageNet images (3 channels)
- First two steerable conv layers downsample the resolution to res_latent x res_latent
  (the code computes suitable integer strides; requires L to be divisible by res_latent)
- Then N steerable layers that preserve resolution (padding chosen accordingly)
- Readout: group pooling + standard conv / pooling to produce 1000 classes

Optional flags are provided to toggle common good-practice components for ablations.

Notes:
- This code targets escnn (QUVA-Lab). It uses the high-level escnn.nn modules: R2Conv,
  FieldType, GeometricTensor, InnerBatchNorm (IIDBatchNorm2d), nonlinearities and GroupPooling.
- If you plan to `export()` this model to pure PyTorch for faster inference, many escnn modules
  implement `export()` and `SequentialModule.export()` (not all modules may support a full export).

"""
from typing import Optional

import torch
import torch.nn as nn

from escnn import gspaces
from escnn import nn as enn


def _get_gspace(group_str: str):
    """Return an escnn gspace object based on the requested group_name.

    Supported names: 'C_1' (trivial), 'C_4' (4-fold rotations),
    'D_1' (reflection only), 'D_4' (rotations+reflections of order 4).
    """
    group_dict = {
        'C_1': gspaces.trivialOnR2(),
        'C_4': gspaces.rot2dOnR2(4),
        'D_1': gspaces.flip2dOnR2(),
        'D_4': gspaces.flipRot2dOnR2(4)
    }
    if group_str not in group_dict:
        raise ValueError(f"Unsupported group: {group_str}. Choose one of C_1, C_4, D_1, D_4")
    return group_dict[group_str]

class ScalableSteerableCNN(nn.Module):
    def __init__(
        self,
        L: int,
        group_name: str,
        n_latent: int,
        num_classes: int = 1000,
        base_width: int = 64,
        activation: str = "relu",  # 'relu' | 'elu' | 'fourier_elu'
        use_batchnorm: bool = True,
        dropout: Optional[float] = None,
        use_residual: bool = False,
        widen_factor: float = 1.0,
        latent_res: int = 14
    ):
        """Construct the steerable CNN.

        Important:
        - L must be divisible by latent_res (so that two downsampling convs can reach exact latent_res x latent_res)
          The code tries to pick integer strides to perform the downsampling in two layers.

        Optional flags (for ablations):
        - use_batchnorm: whether to use equivariant batchnorm (InnerBatchNorm / IIDBatchNorm2d)
        - dropout: dropout probability after GroupPooling (if set)
        - use_residual: add simple residual connections in the n_latent layers
        - activation: choice of equivariant nonlinearity
        - widen_factor: scale the channel counts
        - latent_res: spatial resolution after downsampling (default 14, e.g. 16)
        """
        super().__init__()

        self.L = L
        self.group_name = group_name
        self.n_latent = n_latent
        self.num_classes = num_classes
        self.base_width = int(base_width * widen_factor)
        self.latent_res = latent_res

        gs = _get_gspace(group_name)
        self.gspace = gs

        # input field: trivial representation for RGB channels
        in_type = enn.FieldType(gs, 3 * [gs.trivial_repr])

        # helper to create hidden FieldTypes using the regular representation (a common choice)
        def hidden_type(channels: int):
            # channels is number of scalar fields; choose regular_repr to capture group structure
            return enn.FieldType(gs, [gs.regular_repr] * channels)



        if latent_res == 16 and L == 64:
            # we hardcode this case since this is exactly what we want for the first tests
            print("Using hardcoded strides and channels for L=64, latent_res=16")
            s1, s2 = 2, 2
            pad1, pad2 = 3, 3
            k1, k2 = 7, 7
            c1, c2, c_mid = 32, 16, 32
        else:
            # --- sanity checks and setup
            print("Trying to infer strides and widths. This does not seem to work properly...")
            if L % latent_res != 0:
                raise ValueError(f"Input size L must be divisible by latent_res ({latent_res}) so that two downsampling layers can reach {latent_res}x{latent_res} exactly.")

            # compute downsampling strides for first two layers
            total_down = L // latent_res  # integer
            # Find two integer strides s1 and s2 such that s1 * s2 = total_down
            # Prefer s1 and s2 to be as close as possible
            s1 = int(np.sqrt(total_down))
            while total_down % s1 != 0 and s1 > 1:
                s1 -= 1
            s2 = total_down // s1
            if s1 * s2 != total_down:
                s1 = total_down
                s2 = 1

            # channels per stage (you can adjust these)
            c1 = max(16, self.base_width // 2)
            c2 = self.base_width
            c_mid = self.base_width * 2

        # --- build equivariant backbone
        layers = []

        # First downsampling R2Conv: from RGB trivial fields to c1 regular fields
        out1_type = hidden_type(c1 // gs.fibergroup.size if hasattr(gs, 'fibergroup') and hasattr(gs.fibergroup, 'size') else c1)
        # The above tries to be safe: regular_repr has multiplicity; often `gs.fibergroup.size` equals group order.
        # Simpler approach: set multiplicity = c1. escnn will accept repeated reps.
        out1_type = hidden_type(c1)

        #k1 = 7
        #pad1 = (k1 - s1) // 2
        layers.append(enn.R2Conv(in_type, out1_type, kernel_size=k1, stride=s1, padding=pad1, bias=False))
        if use_batchnorm:
            layers.append(enn.InnerBatchNorm(out1_type))
        layers.append(self._get_activation(out1_type, activation))

        # Second downsampling R2Conv: keep increasing channels and downsample to latent_res x latent_res
        out2_type = hidden_type(c2)
        #k2 = 3
        #pad2 = k2 // 2
        layers.append(enn.R2Conv(out1_type, out2_type, kernel_size=k2, stride=s2, padding=pad2, bias=False))
        if use_batchnorm:
            layers.append(enn.InnerBatchNorm(out2_type))
        layers.append(self._get_activation(out2_type, activation))

        # Now the spatial resolution should be latent_res x latent_res

        # n_latent steerable layers that keep resolution constant
        cur_type = out2_type
        latent_type = hidden_type(c_mid)
        for _ in range(n_latent):
            k = 3
            pad = k // 2
            conv = enn.R2Conv(cur_type, latent_type, kernel_size=k, stride=1, padding=pad, bias=False)

            block = [conv]
            if use_batchnorm:
                block.append(enn.InnerBatchNorm(latent_type))
            block.append(self._get_activation(latent_type, activation))

            # optional residual: simple 1x1 equivariant conv to match channels then add
            if use_residual: # i have not tested these
                # projection for skip connection
                proj = enn.R2Conv(cur_type, latent_type, kernel_size=1, stride=1, padding=0, bias=False)
                block = [enn.SequentialModule(*block)]
                # Wrap into a small residual module using escnn SequentialModule
                res_module = _EquivariantResidual(cur_type, latent_type, proj, block[0])
                layers.append(res_module)
                cur_type = latent_type
            else:
                layers.extend(block)
                cur_type = latent_type

        # Group pooling to get invariant features (collapse group dimension)
        layers.append(enn.GroupPooling(latent_type))

        # After GroupPooling we have a FieldType with trivial representations -> we can export to plain PyTorch
        self.equiv_backbone = enn.SequentialModule(*layers)

        self._equivariant_readout = None  # will be built on first forward
        self._invariant_readout = None  # will be built on first forward

        # store flags
        self.activation = activation
        self.use_batchnorm = use_batchnorm
        self.dropout = dropout
        self.use_residual = use_residual

    def _get_activation(self, field_type: enn.FieldType, activation: str):
        a = activation.lower()
        if a == 'relu':
            return enn.ReLU(field_type, inplace=True)
        elif a == 'elu':
            return enn.ELU(field_type, inplace=True)
        elif a == 'fourier_elu':
            # FourierELU exists in escnn; fall back to ELU if unavailable at runtime
            try:
                return enn.FourierELU(field_type)
            except Exception:
                return enn.ELU(field_type, inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
    def _build_equivariant_readout_stage(self, in_type):
        # Build a small equivariant readout head using escnn modules.
        layers = []
        gs = self.gspace

        # Downsample spatially with equivariant conv and pooling
        out_type1 = enn.FieldType(gs, [gs.regular_repr] * max(128, in_type.size // 2))
        layers.append(enn.R2Conv(in_type, out_type1, kernel_size=3, padding=1, bias=False))
        layers.append(enn.InnerBatchNorm(out_type1))
        layers.append(enn.ReLU(out_type1, inplace=True))
        layers.append(enn.PointwiseAvgPoolAntialiased(out_type1, sigma=0.66, stride=2))  # Downsample spatially

        # Further downsample if needed
        out_type2 = enn.FieldType(gs, [gs.regular_repr] * max(64, out_type1.size // 2))
        layers.append(enn.R2Conv(out_type1, out_type2, kernel_size=3, padding=1, bias=False))
        layers.append(enn.InnerBatchNorm(out_type2))
        layers.append(enn.ReLU(out_type2, inplace=True))
        layers.append(enn.PointwiseAvgPoolAntialiased(out_type2, sigma=0.66, stride=2))

        # Group pooling to get invariance
        layers.append(enn.GroupPooling(out_type2))

        return enn.SequentialModule(*layers)
    
    def _build_invariant_readout_stage(self, in_channels, num_classes, dropout):
        layers = []
        #gs = self.gspace
        # Flatten and final linear layer
        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_channels, num_classes))
        # Optionally add dropout
        if dropout is not None and dropout > 0:
            layers.insert(-1, nn.Dropout(dropout))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        # x shape: (B, 3, L, L)
        # Build GeometricTensor from input and run the equivariant pipeline
        in_type = enn.FieldType(self.gspace, 3 * [self.gspace.trivial_repr])
        x_geo = enn.GeometricTensor(x, in_type)
        z_geo = self.equiv_backbone(x_geo)

        self.device = z_geo.tensor.device

        if self._equivariant_readout is None:
            #num_classes, dropout = self.num_classes, self.dropout
            #z_type = z_geo.type
            self._equivariant_readout = self._build_equivariant_readout_stage(z_geo.type).to(self.device)
            #self._equivariant_readout.to(z_geo.tensor.device)


        #print("z.shape = ", z_geo.shape)
        z_geo = self._equivariant_readout(z_geo)
        z = z_geo.tensor
        #in_channels = z.shape[1]*self.latent_res*self.latent_res
        in_channels = z.shape[1] * self.latent_res // 4 * self.latent_res // 4
        #print("z.shape = ", z.shape)

        if self._invariant_readout is None:
            self._invariant_readout = self._build_invariant_readout_stage(in_channels, self.num_classes, self.dropout).to(self.device)
            #self._invariant_readout.to(z_geo.tensor.device)

        # Convert z to GeometricTensor for equivariant readout
        #gs = self.gspace
        #out_channels = z_geo.shape[1]
        #in_type = enn.FieldType(gs, [gs.regular_repr] * out_channels)
        #z_geo = enn.GeometricTensor(z, in_type)
        out = self._invariant_readout(z)
        return out


class _EquivariantResidual(enn.EquivariantModule):
    """A tiny helper residual module implemented with escnn modules.

    It applies proj(input) and block(input) then sums them. Both inputs must be GeometricTensors with
    compatible shapes and FieldTypes.
    """

    def __init__(self, in_type: enn.FieldType, out_type: enn.FieldType, proj: enn.R2Conv, block: enn.EquivariantModule):
        super().__init__()
        self.in_type = in_type
        self.out_type = out_type
        self.proj = proj
        self.block = block

    def forward(self, input: enn.GeometricTensor):
        s = self.proj(input)
        y = self.block(input)
        return s + y




# Example quick test (run in an environment that has escnn installed):
# model = ScalableSteerableCNN(L=224, group_name='C_4', n_latent=4, base_width=64, use_batchnorm=True, dropout=0.5)
# x = torch.randn(2,3,224,224)
# y = model(x)
# print(y.shape)  # -> (2, 1000)

#if __name__ == '__main__':
#    m = ScalableSteerableCNN(L=224, group_name='C_4', n_latent=4)
#    print(m)



@register_model
def scalablesteerablecnn(pretrained: bool = False, **kwargs) -> ScalableSteerableCNN:
    """Constructs a Scalable Steerable CNN model using layers from the ESCNN package.
    """
    if pretrained:
        raise ValueError("Support for pretrained ESCNN models not yet implemented.")
    print("Creating a Scalable Steerable CNN model using default class constructor. If " \
          "something seems wrong, should probably implement using full " \
            "`build_model_with_cfg` wrapper.")

    # Default arguments for ScalableSteerableCNN
    default_model_args = dict(
        L=64,
        group_name='C_4',
        n_latent=4,
        num_classes=1000,
        base_width=64,
        activation="relu",
        use_batchnorm=True,
        dropout=None,
        use_residual=False,
        widen_factor=1.0,
        latent_res=16
    )
    for key in ['pretrained_cfg', 'pretrained_cfg_overlay', 'cache_dir', 'in_chans', 'drop_rate']:
        kwargs.pop(key, None)
    model = ScalableSteerableCNN(**dict(default_model_args, **kwargs))
    return model




class SteerableConvNeXtIsotropic(nn.Module):
    r""" ConvNeXt
        Adaption of isotropic ConvNeXt to use Steerable CNN layers from the escnn package.
        For isotropic ConNext, see Section 3.3 of https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depth (tuple(int)): Number of blocks. Default: 18.
        dims (int): Feature dimension. Default: 384
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self,
                 group='D_1',
                 in_chans=3,
                 num_classes=1000, 
                 depth=18,
                 dim=384,
                 **kwargs,
                 ):
        super().__init__()

        self.gs = _get_gspace(group)
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.depth = depth
        self.dim = dim


        s1, s2 = 2, 2
        pad1, pad2 = 3, 3
        k1, k2 = 7, 7
        c1, c2 = 32, 64 # here c2 has to be equal to dim, but before setting c2=dim we need to adjust k, pad and s accordingly
        self.embedder = enn.SequentialModule(
            enn.R2Conv(enn.FieldType(self.gs, in_chans * [self.gs.trivial_repr]), 
                       enn.FieldType(self.gs, c1 * [self.gs.regular_repr]), 
                       padding=pad1, kernel_size=k1, stride=s1, bias=False),
            enn.R2Conv(enn.FieldType(self.gs, c1 * [self.gs.regular_repr]),
                          enn.FieldType(self.gs, c2 * [self.gs.regular_repr]), 
                          padding=pad2, kernel_size=k2, stride=s2, bias=False)
        )

        self.blocks = nn.Sequential(*[SteerableConvNeXtBlock(
                                    gs=self.gs,
                                    dim=dim, 
                                    )
                                    for i in range(depth)])

        self.groupnorm = enn.GroupPooling(enn.FieldType(self.gs, [self.gs.regular_repr] * dim))
        self.norm = nn.LayerNorm((dim, ), eps=1e-6)

        self.head = nn.Linear(dim, num_classes)


        # TODO: i removed custom weight init, check if needed

    def forward_features(self, x):
        x = self.embedder(x)
        x = self.blocks(x)
        x = self.groupnorm(x)
        x = x.tensor
        x = x.mean([-2, -1])  # global average pooling, (N, C, H, W) -> (N, C)
        x = self.norm(x)
        return x 

    def forward(self, x):
        x = enn.GeometricTensor(x, enn.FieldType(self.gs, self.in_chans * [self.gs.trivial_repr]))
        x = self.forward_features(x)
        x = self.head(x)
        return x
    


class SteerableConvNeXtBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (1) as we find it faster in PyTorch with torch.compile
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0 (removed)
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6. (removed)
    """
    def __init__(self, gs, dim):
        super().__init__()
        self.gs = gs

        self.dwconv = enn.R2Conv(self.hidden_type(dim), self.hidden_type(dim), kernel_size=7, padding=3, groups=dim, bias=False)

        # The original ConvNeXt has a layernorm here, while I replaced it with an ecnn-batchnorm 
        self.norm = enn.IIDBatchNorm2d(self.hidden_type(dim), eps=1e-6, affine=True)
        # TODO: what to replace these pointwise convs with?
        self.pwconv1 = enn.R2Conv(self.hidden_type(dim), self.hidden_type(4 * dim), kernel_size=1, bias=False)

        self.act = enn.GELU(self.hidden_type(4 * dim))
        self.pwconv2 = enn.R2Conv(self.hidden_type(4 * dim), self.hidden_type(dim), kernel_size=1, bias=False)

        # TODO: replace layer_scale_init_values and/or drop_path with something else?
        #self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((1, dim, 1, 1)), 
        #                            requires_grad=True) if layer_scale_init_value > 0 else None
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def hidden_type(self, channels: int):
        # channels is number of scalar fields; choose regular_repr to capture group structure
        return enn.FieldType(self.gs, [self.gs.regular_repr] * channels)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        # this res connection is regularised in ConvNeXt
        x = input + x
        return x


@register_model
def steerable_convnext_isotropic_small(pretrained=False, **kwargs):
    model = SteerableConvNeXtIsotropic(depth=18, dim=64, **kwargs)
    if pretrained:                                     
        raise NotImplementedError()
    return model
