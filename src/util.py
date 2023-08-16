# adding a dense einsum layer. Note this is very different from how the pytorch og code works

import math
import torch
from torch import nn

_CHR_IDX = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m"]


class DenseEinsum(nn.Module):
    """A densely connected layer that uses tf.einsum as the backing computation.
    This layer can perform einsum calculations of arbitrary dimensionality.
    Arguments:
    output_shape: Positive integer or tuple, dimensionality of the output space.
    num_summed_dimensions: The number of dimensions to sum over. Standard 2D
      matmul should use 1, 3D matmul should use 2, and so forth.
    activation: Activation function to use. If you don't specify anything, no
      activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix.
    bias_initializer: Initializer for the bias vector.
    kernel_regularizer: Regularizer function applied to the `kernel` weights
      matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to the output of the
      layer (its "activation")..
    kernel_constraint: Constraint function applied to the `kernel` weights
      matrix.
    bias_constraint: Constraint function applied to the bias vector.
  Input shape:
    N-D tensor with shape: `(batch_size, ..., input_dim)`. The most common
      situation would be a 2D input with shape `(batch_size, input_dim)`.
  Output shape:
    N-D tensor with shape: `(batch_size, ..., units)`. For instance, for a 2D
      input with shape `(batch_size, input_dim)`, the output would have shape
      `(batch_size, units)`.
  """

    def __init__(self,
               input_shape,
               output_shape,
               num_summed_dimensions=1,
               activation=None,
               use_bias=True,
               kernel_initializer="glorot_uniform",
               bias_initializer="zeros",
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
        super().__init__()

        self._output_shape = output_shape if isinstance(
            output_shape, (list, tuple)) else (output_shape,)
        self._activation = activation
        self._use_bias = use_bias
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer
        self._kernel_constraint = kernel_constraint
        self._bias_constraint = bias_constraint
        self._num_summed_dimensions = num_summed_dimensions
        self._einsum_string = None
        self.input_shape = input_shape

        input_rank = len(input_shape)
        free_input_dims = input_rank - self._num_summed_dimensions
        output_dims = len(self._output_shape)

        self._einsum_string = self._build_einsum_string(free_input_dims,
                                                        self._num_summed_dimensions,
                                                        output_dims)

        # This is only saved for testing purposes.

        self._kernel = torch.Tensor(input_shape[free_input_dims:], self._output_shape)
        self._kernel = nn.Parameter(self._kernel)
        
        if self._use_bias:
            self._bias = torch.Tensor(self._output_shape)
            self._bias = nn.Parameter(self._bias)
        else:
            self._bias = None

# initilaize the weights and bias
        if self.kernel_initializer is None :
          nn.init.kaiming_uniform_(self._kernel, a=math.sqrt(5)) 
        else :
          self.kernel_initializer(self._kernel)
        if self._bias is not None :
          if self._bias_initializer is None :
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self._kernel)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self._bias, -bound, bound)
          else :
              self._bias_initializer(self._bias)

    def cast_inputs(self, inputs):
        pass

    def _build_einsum_string(self, free_input_dims, bound_dims, output_dims):
        input_str = ""
        kernel_str = ""
        output_str = ""
        letter_offset = 0
        for i in range(free_input_dims):
            char = _CHR_IDX[i + letter_offset]
            input_str += char
            output_str += char

        letter_offset += free_input_dims
        for i in range(bound_dims):
            char = _CHR_IDX[i + letter_offset]
            input_str += char
            kernel_str += char

        letter_offset += bound_dims
        for i in range(output_dims):
            char = _CHR_IDX[i + letter_offset]
            kernel_str += char
            output_str += char

        return input_str + "," + kernel_str + "->" + output_str

        
    def forward(self, inputs):
        ret = torch.einsum(self._einsum_string, inputs, self._kernel)
        if self._use_bias:
            ret += self._bias
        if self._activation is not None:
            ret = self._activation(ret)
        return ret