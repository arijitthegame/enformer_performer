# pylint: skip-file

import inspect
from typing import Any, Callable, Dict, Iterable, Optional, Text, Union

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from performer.tf_version import fast_attention

SEQUENCE_LENGTH = 196_608
BIN_SIZE = 128
TARGET_LENGTH = 896

def exponential_linspace_int(start, end, num, divisible_by=1):
  """Exponentially increasing values of integers."""
  def _round(x):
    return int(np.round(x / divisible_by) * divisible_by)

  base = np.exp(np.log(end / start) / (num - 1))
  return [_round(start * base**i) for i in range(num)]

class TargetLengthCrop1D(tf.keras.layers.Layer):
  """Crop sequence to match the desired target length."""

  def __init__(self, target_length: int, name='target_length_crop'):
    super().__init__(name=name)
    self._target_length = target_length

  def call(self, inputs):
    trim = (inputs.shape[-2] - self._target_length) // 2
    if trim < 0:
      raise ValueError('inputs longer than target length')

    return inputs[..., trim:-trim, :]


def pooling_module(kind, pool_size):
  """Pooling module wrapper."""
  if kind == 'attention':
    return SoftmaxPooling1D(pool_size=pool_size, per_channel=True,
                            w_init_scale=2.0)
  elif kind == 'max':
    return tf.keras.layers.MaxPool1D(pool_size=pool_size, padding='same')
  else:
    raise ValueError(f'Invalid pooling kind: {kind}.')


class SoftmaxPooling1D(tf.keras.layers.Layer):
  """Pooling operation with optional weights."""

  def __init__(self,
               pool_size: int = 2,
               per_channel: bool = False,
               w_init_scale: float = 0.0,
               name: str = 'softmax_pooling'):
    """Softmax pooling.
    Args:
      pool_size: Pooling size, same as in Max/AvgPooling.
      per_channel: If True, the logits/softmax weights will be computed for
        each channel separately. If False, same weights will be used across all
        channels.
      w_init_scale: When 0.0 is equivalent to avg pooling, and when
        ~2.0 and `per_channel=False` it's equivalent to max pooling.
      name: Module name.
    """
    super().__init__(name=name)
    self._pool_size = pool_size
    self._per_channel = per_channel
    self._w_init_scale = w_init_scale
    self._logit_linear = None


  def _initialize(self, num_features):
    self._logit_linear = tf.keras.layers.Dense(
        num_features if self._per_channel else 1,
        use_bias=False,  # Softmax is agnostic to shifts.
       kernel_initializer = tf.keras.initializers.Identity(
    gain=1.0))

  def call(self, inputs):
    _, length, num_features = inputs.shape
    self._initialize(num_features)
    inputs = tf.reshape(
        inputs,
        (-1, length // self._pool_size, self._pool_size, num_features))
    return tf.reduce_sum(
        inputs * tf.nn.softmax(self._logit_linear(inputs), axis=-2),
        axis=-2)

class conv_block(tf.keras.layers.Layer):
    def __init__(self, filters: int, width=1, padding = 'same', kernel_initializer = 'glorot_uniform', **kwargs):
        super(conv_block, self).__init__()

        self.filters = filters
        self.width = width
        self.padding = padding
        self.kernel_initializer = kernel_initializer

        self.conv1 = tf.keras.layers.Conv1D(filters=self.filters, kernel_size=self.width, padding=self.padding, kernel_initializer=self.kernel_initializer)
        self.norm = tf.keras.layers.BatchNormalization() 
        self.gelu = tfa.layers.GELU() 

    def call(self, inputs):
        x = self.norm(inputs)
        x = self.gelu(x)
        x = self.conv1(x)
        return x

class stem(tf.keras.layers.Layer):
    def __init__(self, channels: int, width=15, padding = 'same', pool_size = 2, pooling_type : str = 'attention', **kwargs): 
        super(stem, self).__init__()

        self.channels = channels
        self.width = width
        self.padding = padding
        self.pooling_type = pooling_type
        self.pool_size = pool_size

        self.conv1 = tf.keras.layers.Conv1D(filters=self.channels//2, kernel_size=self.width, padding=self.padding)
        self.conv_block = conv_block(self.channels//2, kernel_size=1)
        self.pooling = pooling_module(self.pooling_type, self.pool_size)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = x + self.conv_block(x)
        x = self.pooling(x)
        return x 

class pointwise_conv_block(tf.keras.layers.Layer):
    def __init__(self, filters: int, width=5, padding = 'same', pooling_size = 2, pooling_type : str = 'attention', **kwargs):
        super(pointwise_conv_block, self).__init__()
        self.filters = filters
        self.width = width
        self.padding = padding
        self.pooling_type = pooling_type
        self.pool_size = pooling_size

        self.convolution_block = conv_block(self.filters, kernel_size = self.width, padding=self.padding)
        self.pooling = pooling_module(self.pooling_type, self.pool_size)
        self.pointwise_conv = conv_block(filters=self.filters, kernel_size=1, padding=self.padding)

    def call(self, inputs):
        x = self.convolution_block(inputs)
        x = x + self.pointwise_conv(x)
        x = self.pooling(x)
        return x

    
class conv_tower(tf.keras.layers.Layer):
    def __init__(self, channels: int, num_layers = 6, divisible_by = 128, width=5, padding = 'same', pool_size = 2, pooling_type : str = 'attention', **kwargs):
        super(conv_tower, self).__init__()
        self.channels = channels 
        self.num_layers = num_layers
        self.divisible_by = divisible_by
        self.width = width
        self.padding = padding
        self.pooling_type = pooling_type
        self.pool_size = pool_size


        self.filter_list = exponential_linspace_int(start=self.channels // 2, end=self.channels,
                                           num=self.num_layers, divisible_by=self.divisible_by)
        self.point_wise_conv_block = []
        for i, num_filters in enumerate(self.filter_list):
            self.point_wise_conv_block.append(pointwise_conv_block(num_filters, kernel_size=self.width, padding=self.padding, pooling_size=self.pool_size, pooling_type=self.pooling_type))

    def call(self, inputs):
        x = inputs
        for i, num_filters in enumerate(self.filter_list):
            x = self.point_wise_conv_block[i](x)
        return x

class final_pointwise(tf.keras.layers.Layer):
    def __init__(self, channels: int, dropout : float, padding = 'same', **kwargs):
        super(final_pointwise, self).__init__()
        self.channels = channels
        self.dropout = dropout
        self.padding = padding

        self.conv1 = conv_block(filters=self.channels*2, kernel_size=1, padding=self.padding)
        self.dropout = tf.keras.layers.Dropout(self.dropout/8)
        self.norm = tfa.layers.GELU()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.dropout(x)
        return self.norm(x)


## This is the main class that will be used to create the model. Tested all the above classes and they work as desired.

#TODO: FINISH THIS CLASS
class FastEnformer(tf.keras.Model) :
    def __init__(self, dim, d_model, 
              channels: int = 1536,
               num_transformer_layers: int = 11,
               num_conv_layers: int = 6,
               num_heads: int = 8,
               dropout_rate = .4,
               pooling_type: str = 'attention',
               hidden_size = 256,
               attention_dropout = .2,
               kernel_transformation = 'softmax_kernel_transformation',
               numerical_stabilizer = 0.001,
               causal = False,
               nb_random_features = 64, 
               max_seq_length,
               num_realizations=1,
               norm_layer=None, 
               rel_pos_bins=None, 
               use_spe=False, 
               spe_type=None, 
               kernel_size=None, 
              use_rot_emb = False,
              use_mask_pos = False, 
              normalize = False
    ):
        super(FastEnformer, self).__init__()
        self.channels = channels
        self.num_transformer_layers = num_transformer_layers
        self.num_conv_layers = num_conv_layers
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        self.pooling_type = pooling_type
        self.hidden_size = hidden_size
        self.attention_dropout = attention_dropout
        self.kernel_transformation = kernel_transformation
        self.numerical_stabilizer = numerical_stabilizer
        self.causal = causal
        self.nb_random_features = nb_random_features
        self.dim = dim
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.num_realizations = num_realizations
        self.norm_layer = norm_layer
        self.rel_pos_bins = rel_pos_bins
        self.use_spe = use_spe
        self.spe_type = spe_type
        self.kernel_size = kernel_size
        self.use_rot_emb = use_rot_emb
        self.use_mask_pos = use_mask_pos
        self.normalize = normalize

        self.heads_channels = {'human': 5313, 'mouse': 1643}
        self.early_conv_layer_nums = self.num_conv_layers + 1

        self.stem = stem(self.channels)
        self.conv_tower = conv_tower(self.channels//2, num_layers=self.num_conv_layers)
        self.transformer = PerformerEncoder(num_layers=self.num_transformer_layers, n_heads=self.num_heads, d_model=self.channels, dim = self.dim, \
                                        max_seq_length=SEQUENCE_LENGTH//(2**(self.early_conv_layer_nums)), nb_random_features=self.nb_random_features, rel_pos_bins=self.rel_pos_bins, \
                                        use_spe=self.use_spe, spe_type=self.spe_type, kernel_size=self.kernel_size, use_rot_emb=self.use_rot_emb, use_mask_pos=self.use_mask_pos, normalize=self.normalize)
        self.crop_final = TargetLengthCrop1D(TARGET_LENGTH, name='target_input')
        self.final_pointwise = final_pointwise(self.channels, dropout=self.dropout_rate)
      
        with tf.name_scope('heads'):
            self._heads = {
             head: tf.keras.Sequential(
              [tf.keras.layers.Dense(num_channels, activation='softplus')],
              name=f'head_{head}')
          for head, num_channels in self.heads_channels.items()
      }

    def call(self, inputs, training:bool = True):
        x = self.stem(inputs)
        x = self.conv_tower(x)
        x = self.transformer(x, training=training)
        x = self.crop_final(x)
        x = self.final_pointwise(x)
    
        return {
        head: head_module(x, training=training)
        for head, head_module in self._heads.items()
    }
 