import torch.nn as nn

normalization_choices = {
    'none': None,
    'batchConv': nn.BatchNorm2d,
    'batchFC': nn.BatchNorm1d,
    'layer': lambda num_features: nn.GroupNorm(1, num_features),  # LayerNorm for convs
    'instanceConv': nn.InstanceNorm2d,
    'instanceFC': nn.InstanceNorm1d,
    'group2': lambda num_features: nn.GroupNorm(num_groups=2, num_channels=num_features),
    'group4': lambda num_features: nn.GroupNorm(num_groups=4, num_channels=num_features),
    'group6': lambda num_features: nn.GroupNorm(num_groups=6, num_channels=num_features),
}

