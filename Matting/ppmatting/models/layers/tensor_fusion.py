import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from  models.models_seg import layers

from ppmatting.models.layers import tensor_fusion_helper as helper


class MLFF(nn.Layer):
    """
    Multi-level features are fused adaptively by obtaining spatial attention.

    Args:
        in_channels(list): The channels of input tensors.
        mid_channles(list): The middle channels while fusing the features.
        out_channel(int): The output channel after fusing.
        merge_type(str): Which type to merge the multi features before output. 
            It should be one of ('add', 'concat'). Default: 'concat'.
    """

    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channel,
                 merge_type='concat'):
        super().__init__()

        self.merge_type = merge_type

        # Check arguments
        if len(in_channels) != len(mid_channels):
            raise ValueError(
                "`mid_channels` should have the same length as `in_channels`, but they are {} and {}".
                format(mid_channels, in_channels))
        if self.merge_type == 'add' and len(np.unique(np.array(
                mid_channels))) != 1:
            raise ValueError(
                "if `merge_type='add', `mid_channels` should be same of all input features, but it is {}.".
                format(mid_channels))

        self.pwconvs = nn.ModuleList()
        self.dwconvs = nn.ModuleList()
        for in_channel, mid_channel in zip(in_channels, mid_channels):
            self.pwconvs.append(
                layers.ConvBN(
                    in_channel, mid_channel, 1, bias_attr=False))
            self.dwconvs.append(
                layers.ConvBNReLU(
                    mid_channel,
                    mid_channel,
                    3,
                    padding=1,
                    groups=mid_channel,
                    bias_attr=False))

        num_feas = len(in_channels)
        self.conv_atten = nn.Sequential(
            layers.ConvBNReLU(
                2 * num_feas,
                num_feas,
                kernel_size=3,
                padding=1,
                bias_attr=False),
            layers.ConvBN(
                num_feas, num_feas, kernel_size=3, padding=1, bias_attr=False))

        if self.merge_type == 'add':
            in_chan = mid_channels[0]
        else:
            in_chan = sum(mid_channels)
        self.conv_out = layers.ConvBNReLU(
            in_chan, out_channel, kernel_size=3, padding=1, bias_attr=False)

    def forward(self, inputs, shape):
        """
        args:
            inputs(list): List of tensor to be fused.
            shape(Tensor): A tensor with two elements like (H, W).
        """
        feas = []
        for i, input in enumerate(inputs):
            x = self.pwconvs[i](input)
            x = F.interpolate(
                x, size=shape, mode='bilinear', align_corners=False)
            x = self.dwconvs[i](x)
            feas.append(x)

        atten = helper.avg_max_reduce_channel(feas)
        atten = F.sigmoid(self.conv_atten(atten))

        feas_att = []
        for i, fea in enumerate(feas):
            fea = fea * (atten[:, i, :, :].unsqueeze(1))
            feas_att.append(fea)
        if self.merge_type == 'concat':
            out = torch.cat(feas_att, dim=1)
        else:
            out = sum(feas_att)

        out = self.conv_out(out)
        return out
