"""DenseNet for molecule"""

import torch
from torch import nn
from torch.utils import checkpoint as cp

from deep_scaffold.layers.casual import CausalMolConvBlock
from deep_scaffold.layers.mol_conv import MolConv
from deep_scaffold.layers.utils import BNReLULinear

__all__ = ['DenseNet']


def _bn_function_factory(bn_module):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, -1)
        bottleneck_output = bn_module(concated_features)
        return bottleneck_output

    return bn_function


class DenseLayer(nn.Module):
    """Building block for DenseNet"""

    def __init__(self, num_atom_features, num_bond_types, num_bn_features, num_out_features, efficient=False,
                 activation='elu', conditional=False, num_cond_features=None, activation_cond=None):
        """Constructor

        Args:
            num_atom_features (int): The number of input features for each atom
            num_bond_types (int): The number of bond types considered
            num_bn_features (int): The number of features for the bottlenec layer
            num_out_features (int): The number of output features
            efficient (bool): Whether to use the memory efficient version of densenet, default to False
            activation (str): The type of activation function used
            conditional (bool): Whether to include conditional input, default to False
            num_cond_features (int): The size of conditional input, should be None if self.conditional is False
            activation_cond (str or None): activation function used for conditional input should be None
                                           if self.conditional is False
        """
        super(DenseLayer, self).__init__()

        self.num_atom_features = num_atom_features
        self.num_bond_types = num_bond_types
        self.num_bn_features = num_bn_features
        self.num_out_features = num_out_features
        self.efficient = efficient
        self.activation = activation
        self.conditional = conditional
        self.num_cond_features = num_cond_features
        self.activation_cond = activation_cond

        # modules
        self.bottlenec = BNReLULinear(in_features=self.num_atom_features,
                                      out_features=self.num_bn_features,
                                      activation=self.activation)
        self.conv = MolConv(num_atom_features=self.num_bn_features,
                            num_bond_types=self.num_bond_types,
                            num_out_features=num_out_features,
                            activation=self.activation,
                            conditional=self.conditional,
                            num_cond_features=self.num_cond_features,
                            activation_cond=self.activation_cond)

    def forward(self, atom_features_list, bond_info, cond_features=None):
        """
        Args:
            atom_features_list (list[torch.Tensor]): Input features from previous dense layers for each node
                                                     size=[num_nodes, num_node_features]
            bond_info (torch.Tensor): Bond type information packed into a single matrix
                                      type: torch.long, shape: [-1, 3], where 3 = begin_ids + end_ids + bond_type
            cond_features (torch.Tensor or None): Input conditional features should be None if self.conditional is False

        Returns:
            torch.Tensor: Output feature for each node, size=[num_nodes, hidden_sizes[-1]]
        """
        bn_fn = _bn_function_factory(self.bottlenec)
        if self.efficient and all([atom_features_i.requires_grad for atom_features_i in atom_features_list]):
            atom_features = cp.checkpoint(bn_fn, *atom_features_list)
        else:
            atom_features = bn_fn(*atom_features_list)

        return self.conv(atom_features, bond_info, cond_features)


class DenseNet(nn.Module):
    """Molecular DenseNet"""

    def __init__(self, num_atom_features, num_bond_types, causal_hidden_sizes, num_bn_features, num_k_features,
                 num_layers, num_output_features, efficient=False, activation='elu', conditional=False,
                 num_cond_features=None, activation_cond=None):
        """Molecular DenseNet

        Args:
            num_atom_features (int): The number of input features for each atom
            num_bond_types (int): The number of bond types considered
            causal_hidden_sizes (tuple[int]): The hidden sizes for the preceding causal layers
            num_bn_features (int): The number of features for the bottlenec layer
            num_k_features (int): The output feature for each dense layer
            num_layers (int): The number of dense layers used in this network
            num_output_features (int): The number of output features for each atom
            efficient (bool): Whether to use the memory efficient version of densenet, default to False
            activation (str): The type of activation function used
            conditional (bool): Whether to include conditional input, default to False
            num_cond_features (int): The size of conditional input, should be None if self.conditional is False
            activation_cond (str or None): activation function used for conditional input should be None
                                           if self.conditional is False
        """
        super(DenseNet, self).__init__()

        self.num_atom_features = num_atom_features
        self.num_bond_types = num_bond_types
        self.causal_hidden_sizes = list(causal_hidden_sizes)
        self.num_bn_features = num_bn_features
        self.num_k_features = num_k_features
        self.num_layers = num_layers
        self.num_output_features = num_output_features
        self.efficient = efficient
        self.activation = activation
        self.conditional = conditional
        self.num_cond_features = num_cond_features
        self.activation_cond = activation_cond

        # modules
        self.causal_conv = CausalMolConvBlock(num_atom_features=self.num_atom_features,
                                              num_bond_types=self.num_bond_types,
                                              hidden_sizes=self.causal_hidden_sizes,
                                              activation=self.activation,
                                              conditional=self.conditional,
                                              num_cond_features=self.num_cond_features,
                                              activation_cond=self.activation_cond)
        dense_layers = []
        for i in range(self.num_layers):
            dense_layer = DenseLayer(num_atom_features=(self.causal_hidden_sizes[-1] + i * self.num_k_features),
                                     num_bond_types=self.num_bond_types,
                                     num_bn_features=self.num_bn_features,
                                     num_out_features=self.num_k_features,
                                     efficient=self.efficient,
                                     activation=self.activation,
                                     conditional=self.conditional,
                                     num_cond_features=self.num_cond_features,
                                     activation_cond=self.activation_cond)
            dense_layers.append(dense_layer)

        self.dense_layers = nn.ModuleList(dense_layers)

        self.output = BNReLULinear(in_features=(self.causal_hidden_sizes[-1] + self.num_layers * self.num_k_features),
                                   out_features=self.num_output_features,
                                   activation=self.activation)

    def forward(self, atom_features, bond_info, cond_features=None):
        """
        Args:
            atom_features (torch.Tensor): Input features for each node, size=[num_nodes, num_node_features]
            bond_info (torch.Tensor): Bond type information packed into a single matrix
                                      type: torch.long, shape: [-1, 3], where 3 = begin_ids + end_ids + bond_type
            cond_features (torch.Tensor or None): Input conditional features should be None if self.conditional is False

        Returns:
            torch.Tensor: Output feature for each node, size=[num_nodes, hidden_sizes[-1]]
        """
        atom_features = self.causal_conv(atom_features, bond_info, cond_features)
        atom_features_list = [atom_features, ]
        for dense_layer in self.dense_layers:
            atom_features_i = dense_layer(atom_features_list, bond_info, cond_features)
            atom_features_list.append(atom_features_i)

        atom_features_cat = torch.cat(atom_features_list, dim=-1)
        return self.output(atom_features_cat)
