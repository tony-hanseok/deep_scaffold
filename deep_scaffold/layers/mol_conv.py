"""Some basic layers for molecule convolution"""

import torch
from torch import nn

from deep_scaffold.layers.utils import BNReLULinear

__all__ = ['MolConv']


class MolConv(nn.Module):
    """ Molecule convolution
    Implements the basic unit for molecule convolution network
    """

    def __init__(self, num_atom_features, num_bond_types, num_out_features, activation='elu',
                 conditional=False, num_cond_features=None, activation_cond=None):
        """The graph convolution layer

        Args:
            num_atom_features (int): The number of input features for each node
            num_bond_types (int): The number of bond types
            num_out_features (int): The number of output features for each node
            activation (str or None): The activation type, None if no activation is used
            conditional (bool): Whether to include conditional input, default to False
            num_cond_features (int): The size of conditional input, should be None if self.conditional is False
            activation_cond (str or None): Activation function used for conditional input, should be None
                                           if self.conditional is False
        """
        super(MolConv, self).__init__()
        self.num_atom_features = num_atom_features
        self.num_bond_types = num_bond_types
        self.num_out_features = num_out_features
        self.activation = activation
        self.conditional = conditional
        self.num_cond_features = num_cond_features
        self.activation_cond = activation_cond

        # submodules
        _in_features = self.num_atom_features
        _out_features = self.num_out_features * self.num_bond_types
        if self.activation is not None:
            self.linear = BNReLULinear(in_features=_in_features,
                                       out_features=_out_features,
                                       activation=self.activation)
        else:
            self.linear = nn.Linear(in_features=_in_features, out_features=_out_features)

        # conditional convolution
        self.linear_cond = None
        if self.conditional:
            assert self.num_cond_features is not None and self.activation_cond is not None
            self.linear_cond = BNReLULinear(in_features=self.num_cond_features,
                                            out_features=self.num_out_features,
                                            activation=self.activation_cond)
        else:
            assert self.num_cond_features is None
            assert self.activation_cond is None

    def forward(self, atom_features, bond_info, cond_features=None):
        """The forward function

        Args:
            atom_features (torch.Tensor): Input features for each atom,
                                          type: torch.float32, size: [-1, num_atom_features]
            bond_info (torch.Tensor): Bond information as a sparse matrix
                                      type: torch.float32, dense size: [-1, -1]
            cond_features (torch.Tensor or None): Input conditional features should be None if self.conditional is False

        Returns:
            torch.Tensor: atom features after graph convolution
        """
        # perform linear transformation
        # size: [-1, num_bond_types * num_out_features]
        atom_features = self.linear(atom_features)

        # move num_bond_types to the first dimension
        # size: [-1, num_out_features]
        atom_features = atom_features.view(-1, self.num_bond_types, self.num_out_features)
        atom_features = atom_features.transpose(0, 1).contiguous()
        atom_features = atom_features.view(-1, self.num_out_features)

        if self.conditional:
            assert cond_features is not None
            cond_features = self.linear_cond(cond_features)
            atom_features = torch.cat([atom_features, cond_features], dim=0)
        else:
            assert cond_features is None
        adj = bond_info

        # perform sparse multiplication
        output = torch.spmm(adj, atom_features)

        return output
