"""Implements casual molecule convlutional block"""
import typing as t

import torch
from torch import nn

from deep_scaffold.layers.mol_conv import MolConv

__all__ = ['CausalMolConvBlock']


class CausalMolConvBlock(nn.Module):
    """The causal mol conv block"""

    def __init__(self, num_atom_features, num_bond_types, hidden_sizes, activation='elu', conditional=False,
                 num_cond_features=None, activation_cond=None):
        """The constructor

        Args:
            num_atom_features (int): The number of input features for each node
            num_bond_types (int): The number of bond types considered
            hidden_sizes (t.Iterable): The hidden size and output size for each weave layer
            activation (str): The type of activation unit to use in this module, default to elu
            conditional (bool): Whether to include conditional input, default to False
            num_cond_features (int): The size of conditional input, should be None if self.conditional is False
            activation_cond (str or None): activation function used for conditional input, should be None
                                           if self.conditional is False
        """
        super(CausalMolConvBlock, self).__init__()

        self.num_node_features = num_atom_features
        self.num_bond_types = num_bond_types
        self.hidden_sizes = list(hidden_sizes)
        self.activation = activation
        self.conditional = conditional
        self.num_cond_features = num_cond_features
        self.activation_cond = activation_cond

        layers = []
        in_features_list = [self.num_node_features, ] + list(self.hidden_sizes)[:-1]
        out_features_list = self.hidden_sizes
        for i, (in_features, out_features) in enumerate(zip(in_features_list, out_features_list)):
            if i == 0:
                activation = None
            else:
                activation = self.activation

            layer = MolConv(num_atom_features=in_features,
                            num_bond_types=self.num_bond_types,
                            num_out_features=out_features,
                            activation=activation,
                            conditional=self.conditional,
                            num_cond_features=num_cond_features,
                            activation_cond=self.activation_cond)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

    def forward(self, atom_features, bond_info, cond_features=None):
        """
        Args:
            atom_features (torch.Tensor): Input features for each node, size=[num_nodes, num_node_features]
            bond_info (torch.Tensor): Bond type information packed into a single matrix,
                                      type: torch.long, shape: [-1, 3], where 3 = begin_ids + end_ids + bond_type
            cond_features (torch.Tensor or None): Input conditional features, should be None
                                                  if self.conditional is False

        Returns:
            torch.Tensor: Output feature for each node, size=[num_nodes, hidden_sizes[-1]]
        """
        atom_features_out = atom_features
        for layer in self.layers:
            atom_features_out = layer(atom_features_out, bond_info, cond_features)
        return atom_features_out
