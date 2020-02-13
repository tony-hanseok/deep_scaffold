"""
Shared utility layers
"""
from torch import nn

from deep_scaffold import ops

__all__ = ['BNReLULinear']


class BNReLULinear(nn.Module):
    """Linear layer with bn->relu->linear architecture"""

    def __init__(self, in_features, out_features, activation='elu'):
        """The initializer

        Args:
            in_features (int): The number of input features
            out_features (int): The number of output features
            activation (str): The type of activation unit to use in this module, default to elu
        """
        super(BNReLULinear, self).__init__()
        self.bn_relu_linear = nn.Sequential(nn.BatchNorm1d(in_features),
                                            ops.get_activation(activation, inplace=True),
                                            nn.Linear(in_features, out_features, bias=False))

    def forward(self, x):
        """The forward method"""
        return self.bn_relu_linear(x)
