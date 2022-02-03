"""Tensorflow only implementation of Enformer(avsec et al. 2021) using Performer attention
"""

import inspect
from typing import Any, Callable, Dict, Optional, Text, Union, Iterable

import tensorflow.experimental.numpy as tnp
#tnp.experimental_enable_numpy_behavior(prefer_float32=True)
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from tensorflow.keras import layers as kl


from attention_module_tf import MultiheadSelfAttention 

SEQUENCE_LENGTH = 196608
BIN_SIZE = 128
TARGET_LENGTH = 896

class Enformer(tf.keras.Model):
    def __init__(self,
               channels: int = 1536,
               num_transformer_layers: int = 11,
               out_heads: dict = {'human': 5313,
                                  'mouse': 1643},
               num_heads: int = 8,
               pooling_type: str = 'attention',
               name: str = 'enformer_vanilla'):
    
        super().__init__(name=name)
    
        heads_channels = out_heads
        dropout_rate = 0.4
        assert channels % num_heads == 0, ('channels needs to be divisible ' f'by {num_heads}')
    
        whole_attention_kwargs = {
            'attention_dropout_rate': 0.05,
            'initializer': None,
            'key_size': 64,
            'num_heads': 8,
            'num_relative_position_features': channels // num_heads,
            'positional_dropout_rate': 0.01,
            'relative_position_functions': [
                'positional_features_exponential',
                'positional_features_central_mask',
                'positional_features_gamma'
            ],
            'relative_positions': True,
            'scaling': True,
            'value_size': channels // num_heads,
            'zero_initialize': True
        }

    ### defining trunk
    
        trunk_name_scope = tf.name_scope('trunk')
        trunk_name_scope.__enter__()

        def conv_block(filters, width=1, w_init=None, name='conv_block', **kwargs):
            return tf.keras.Sequential([
                    kl.BatchNormalization(axis=-1,
                                          momentum = 0.9,
                                          center=True,
                                          scale=True,
                                          beta_initializer="zeros",
                                          gamma_initializer="ones",
                                          **kwargs),
                    GELU(),
                    kl.Conv1D(filters,
                                width,
                                padding='same',
                                kernel_initializer=w_init, **kwargs),

            ], name=name)

        filter_list = exponential_linspace_int(start=channels // 2, end=channels,
                                       num=6, divisible_by=128)


        stem = tf.keras.Sequential([
                kl.Conv1D(channels // 2, 15,padding='same'),
                Residual(conv_block(channels // 2, 1, name='pointwise_conv_block')),
                pooling_module(pooling_type, pool_size=2)
        ], name='stem')

        conv_tower = tf.keras.Sequential([
                            tf.keras.Sequential([
                            conv_block(num_filters, 5),
                            Residual(conv_block(num_filters, 1, name='pointwise_conv_block')),
                            pooling_module(pooling_type, pool_size=2),
                            ],name=f'conv_tower_block_{i}')
                            for i, num_filters in enumerate(filter_list)
        ], name='conv_tower')

        #pos_encoding = positional_encoding_module(pos_enc_type, name = 'positional_encoding')
        #pos_encoding_dropout = keras.layers.Dropout(positional_dropout_rate)

        def res_transformer_mlp():
            return Residual(tf.keras.Sequential([
                    kl.LayerNormalization(axis=-1, 
                                            scale=True, 
                                            center=True,
                                            beta_initializer="zeros",
                                            gamma_initializer="ones"),
                    kl.Dense(channels * 2),
                    kl.Dropout(dropout_rate),
                    kl.Lambda(lambda x: tf.math.maximum(x, 0.0)),
                    kl.Dense(channels),
                    kl.Dropout(dropout_rate)], name='mlp'))

        transformer = tf.keras.Sequential([
                        tf.keras.Sequential([
                            Residual(tf.keras.Sequential([
                                        kl.LayerNormalization(axis=-1,
                                                                scale=True, 
                                                                center=True,
                                                                beta_initializer="zeros",
                                                                gamma_initializer="ones"),
                                        ### add positional info in every transformer loop
                                #####
                                        MultiheadSelfAttention(**whole_attention_kwargs,
                                                                    name=f'attention_{i}'),
                                        kl.Dropout(dropout_rate)], name='mha'),name='res_trans_1'),
                                        res_transformer_mlp()],name=f'transformer_block_{i}')
                        for i in range(num_transformer_layers)], name='transformer')

        crop_final = TargetLengthCrop1D(TARGET_LENGTH, name='target_input')


        final_pointwise = tf.keras.Sequential([
            conv_block(channels * 2, 1),
            kl.Dropout(dropout_rate / 8),
            GELU()], name='final_pointwise')

        self._trunk = tf.keras.Sequential([stem,
                                        conv_tower,
                                        transformer,
                                        crop_final,
                                        final_pointwise],
                                        name='trunk')
        trunk_name_scope.__exit__(None, None, None)


        with tf.name_scope('heads'):
            self._heads = {
                head: tf.keras.Sequential([
                        kl.Dense(num_channels, activation='softplus')], name=f'head_{head}')
                for head, num_channels in heads_channels.items()
            }
        # pylint: enable=g-complex-comprehension,g-long-lambda,cell-var-from-loop

    @property
    def trunk(self):
        return self._trunk

    @property
    def heads(self):
        return self._heads

    def call(self, inputs, *args, **kwargs) -> Dict[str, tf.Tensor]:
        trunk_embedding = self.trunk(inputs, *args, **kwargs)
        return {
            head: head_module(trunk_embedding)
            for head, head_module in self.heads.items()
        }

    def build_graph(self, input_shape):
        x = tf.keras.layers.Input(shape=(None, input_shape))
        return tf.keras.Model(inputs=[x], 
                              outputs=self.call(x))
    
    @tf.function(input_signature=[
        tf.TensorSpec([None, 131072, 4], tf.float32)])
    def predict_on_batch(self, x):
        """Method for SavedModel."""
        return self(x, training=False)
        
class Residual(kl.Layer):
    def __init__(self, layer: kl.Layer, name: str= 'residual'):
        super().__init__(name=name)
        self._layer = layer

    def call(self, inputs, *args, **kwargs):
        return inputs + self._layer(inputs, *args, **kwargs)


def pooling_module(kind, pool_size):
    """Pooling module wrapper."""
    if kind == 'attention':
        return SoftmaxPooling1D(pool_size=pool_size, per_channel=True,
                            w_init_scale=2.0)
    elif kind == 'max':
        return kl.MaxPool1D(pool_size=pool_size, padding='same')
    else:
        raise ValueError(f'Invalid pooling kind: {kind}.')


class SoftmaxPooling1D(kl.Layer):
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
        
    def get_config(self):
        config = super(SoftmaxPooling1D, self).get_config()
        config.update({"_pool_size": self._pool_size,
                       "_per_channel": self._per_channel,
                       "_w_init_scale": self._w_init_scale})
        return config        

    def build(self, input_shape):
        num_features = input_shape[-1]
        self._logit_linear = kl.Dense(num_features,
                                      use_bias=False,
                                      kernel_initializer=tf.keras.initializers.Identity(self._w_init_scale))
        super(SoftmaxPooling1D, self).build(input_shape)
    

    def call(self, inputs):
        _, length, num_features = inputs.shape
        inputs = tf.reshape(inputs,
                            (-1, length // self._pool_size,
                             self._pool_size, num_features))
        return tf.reduce_sum(inputs * tf.nn.softmax(self._logit_linear(inputs), 
                                                    axis=-2), axis=-2)

    

def one_hot_encode(sequence: str,
                   alphabet: str = 'ACGT',
                   neutral_alphabet: str = 'N',
                   neutral_value: Any = 0,
                   dtype=tnp.float32) -> tnp.ndarray:
    """One-hot encode sequence."""
    def to_uint8(string):
        return tnp.frombuffer(string.encode('ascii'), dtype=tnp.uint8)
    hash_table = tnp.zeros((tnp.iinfo(tnp.uint8).max, len(alphabet)), dtype=dtype)
    hash_table[to_uint8(alphabet)] = tnp.eye(len(alphabet), dtype=dtype)
    hash_table[to_uint8(neutral_alphabet)] = neutral_value
    hash_table = hash_table.astype(dtype)
    return hash_table[to_uint8(sequence)]

def exponential_linspace_int(start, end, num, divisible_by=1):
    """Exponentially increasing values of integers."""
    def _round(x):
        return tf.cast((tf.math.round(x / divisible_by) * divisible_by),dtype=tf.int32)

    base = tf.cast(tnp.exp(tnp.log(end / start) / (num - 1)), dtype=tf.float32)
    return [_round(start * base**i) for i in range(num)]

class TargetLengthCrop1D(keras.layers.Layer):
    """Crop sequence to match the desired target length."""

    def __init__(self, target_length: int, name='target_length_crop'):
        super().__init__(name=name)
        self._target_length = target_length
        
    def get_config(self):
        config = super(TargetLengthCrop1D, self).get_config()
        config.update({"target_length": self._target_length})
        return config

    def call(self, inputs):
        trim = (inputs.shape[-2] - self._target_length) // 2
        if trim < 0:
            raise ValueError('inputs longer than target length')
    #print(inputs[..., trim:-trim, :].shape)
        return inputs[..., trim:-trim, :]


class GELU(kl.Layer):
    def __init__(self, name: str = 'GELU'):
        super().__init__(name=name)
    def get_config(self):
        config = super().get_config()
        return config
    
    def cast_inputs(self, inputs):
       # Casts to float16, the policy's lowest-precision dtype
        return self._mixed_precision_policy.cast_to_lowest(inputs)
    def call(self, x):
        return tf.nn.sigmoid(1.702 * x) * x