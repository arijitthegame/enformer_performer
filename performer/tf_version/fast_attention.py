# pylint: skip-file

import math
import numpy as np
import tensorflow as tf
import util
from spe_tf import *
from einops import rearrange, repeat
from functools import partial

BIG_CONSTANT = 1e8


def create_projection_matrix(m, d, seed=0, scaling=0, struct_mode=False):
  r"""Constructs the matrix of random projections.

  Constructs a matrix of random orthogonal projections. Each projection vector
  has direction chosen uniformly at random and either deterministic length
  \sqrt{d} or length taken from the \chi(d) distribution (in the latter case
  marginal distributions of the projections are d-dimensional Gaussian vectors
  with associated identity covariance matrix).

  Args:
    m: number of random projections.
    d: dimensionality of each random projection.
    seed: random seed used to construct projections.
    scaling: 1 if all the random projections need to be renormalized to have
      length \sqrt{d}, 0 if the lengths of random projections should follow
      \chi(d) distribution.
    struct_mode: if True then products of Givens rotations will be used to
      construct random orthogonal matrix. This bypasses Gram-Schmidt
      orthogonalization.

  Returns:
    The matrix of random projections of the shape [m, d].
  """
  nb_full_blocks = int(m / d)
  block_list = []
  current_seed = seed
  for _ in range(nb_full_blocks):
    if struct_mode:
      q = create_products_of_givens_rotations(d, seed)
    else:
      unstructured_block = tf.random.normal((d, d), seed=current_seed)
      q, _ = tf.linalg.qr(unstructured_block)
      q = tf.transpose(q)
    block_list.append(q)
    current_seed += 1
  remaining_rows = m - nb_full_blocks * d
  if remaining_rows > 0:
    if struct_mode:
      q = create_products_of_givens_rotations(d, seed)
    else:
      unstructured_block = tf.random.normal((d, d), seed=current_seed)
      q, _ = tf.linalg.qr(unstructured_block)
      q = tf.transpose(q)
    block_list.append(q[0:remaining_rows])
  final_matrix = tf.experimental.numpy.vstack(block_list)
  current_seed += 1

  if scaling == 0:
    multiplier = tf.norm(tf.random.normal((m, d), seed=current_seed), axis=1)
  elif scaling == 1:
    multiplier = tf.math.sqrt(float(d)) * tf.ones((m))
  else:
    raise ValueError("Scaling must be one of {0, 1}. Was %s" % scaling)

  return tf.linalg.matmul(tf.linalg.diag(multiplier), final_matrix)


def create_products_of_givens_rotations(dim, seed):
  r"""Constructs a 2D-tensor which is a product of Givens random rotations.

  Constructs a 2D-tensor of the form G_1 * ... * G_k, where G_i is a Givens
  random rotation. The resulting tensor mimics a matrix taken uniformly at
  random form the orthogonal group.

  Args:
    dim: number of rows/columns of the resulting 2D-tensor.
    seed: random seed.

  Returns:
    The product of Givens random rotations.
  """
  nb_givens_rotations = dim * int(math.ceil(math.log(float(dim))))
  q = np.eye(dim, dim)
  np.random.seed(seed)
  for _ in range(nb_givens_rotations):
    random_angle = math.pi * np.random.uniform()
    random_indices = np.random.choice(dim, 2)
    index_i = min(random_indices[0], random_indices[1])
    index_j = max(random_indices[0], random_indices[1])
    slice_i = q[index_i]
    slice_j = q[index_j]
    new_slice_i = math.cos(random_angle) * slice_i + math.sin(
        random_angle) * slice_j
    new_slice_j = -math.sin(random_angle) * slice_i + math.cos(
        random_angle) * slice_j
    q[index_i] = new_slice_i
    q[index_j] = new_slice_j
  return tf.cast(tf.constant(q), dtype=tf.float32)


def relu_kernel_transformation(data,
                               is_query,
                               projection_matrix=None,
                               numerical_stabilizer=0.001):
  """Computes features for the ReLU-kernel.

  Computes random features for the ReLU kernel from
  https://arxiv.org/pdf/2009.14794.pdf.

  Args:
    data: input data tensor of the shape [B, L, H, D], where: B - batch
      dimension, L - attention dimensions, H - heads, D - features.
    is_query: indicates whether input data is a query oor key tensor.
    projection_matrix: random Gaussian matrix of shape [M, D], where M stands
      for the number of random features and each D x D sub-block has pairwise
      orthogonal rows.
    numerical_stabilizer: small positive constant for numerical stability.

  Returns:
    Corresponding kernel feature map.
  """
  del is_query
  if projection_matrix is None:
    return tf.nn.relu(data) + numerical_stabilizer
  else:
    ratio = 1.0 / tf.math.sqrt(
        tf.dtypes.cast(projection_matrix.shape[0], tf.float32))
    data_dash = ratio * tf.einsum("blhd,md->blhm", data, projection_matrix)
    return tf.nn.relu(data_dash) + numerical_stabilizer


def softmax_kernel_transformation(data,
                                  is_query,
                                  projection_matrix,
                                  numerical_stabilizer=0.000001):
  """Computes random features for the softmax kernel using FAVOR+ mechanism.

  Computes random features for the softmax kernel using FAVOR+ mechanism from
  https://arxiv.org/pdf/2009.14794.pdf.

  Args:
    data: input data tensor of the shape [B, L, H, D], where: B - batch
      dimension, L - attention dimensions, H - heads, D - features.
    is_query: indicates whether input data is a query oor key tensor.
    projection_matrix: random Gaussian matrix of shape [M, D], where M stands
      for the number of random features and each D x D sub-block has pairwise
      orthogonal rows.
    numerical_stabilizer: small positive constant for numerical stability.

  Returns:
    Corresponding kernel feature map.
  """
  #changed the projection_matrix to not none
  data_normalizer = 1.0 / (
      tf.math.sqrt(tf.math.sqrt(tf.dtypes.cast(data.shape[-1], tf.float32))))
  data = data_normalizer * data
  ratio = 1.0 / tf.math.sqrt(
      tf.dtypes.cast(projection_matrix.shape[0], tf.float32))
  data_dash = tf.einsum("blhd,md->blhm", data, projection_matrix)
  diag_data = tf.math.square(data)
  diag_data = tf.math.reduce_sum(
      diag_data, axis=tf.keras.backend.ndim(data) - 1)
  diag_data = diag_data / 2.0
  diag_data = tf.expand_dims(diag_data, axis=tf.keras.backend.ndim(data) - 1)
  last_dims_t = (len(data_dash.shape) - 1,)
  attention_dims_t = (len(data_dash.shape) - 3,)
  if is_query:
    data_dash = ratio * (
        tf.math.exp(data_dash - diag_data - tf.math.reduce_max(
            data_dash, axis=last_dims_t, keepdims=True)) + numerical_stabilizer)
  else:
    data_dash = ratio * (
        tf.math.exp(data_dash - diag_data - tf.math.reduce_max(
            data_dash, axis=last_dims_t + attention_dims_t, keepdims=True)) +
        numerical_stabilizer)

  return data_dash


def noncausal_numerator(qs, ks, vs):
  """Computes not-normalized FAVOR noncausal attention AV.

  Args:
    qs: query_prime tensor of the shape [L,B,H,M].
    ks: key_prime tensor of the shape [L,B,H,M].
    vs: value tensor of the shape [L,B,H,D].

  Returns:
    Not-normalized FAVOR noncausal attention AV.
  """
  kvs = tf.einsum("lbhm,lbhd->bhmd", ks, vs)
  return tf.einsum("lbhm,bhmd->lbhd", qs, kvs)


def noncausal_denominator(qs, ks):
  """Computes FAVOR normalizer in noncausal attention.

  Args:
    qs: query_prime tensor of the shape [L,B,H,M].
    ks: key_prime tensor of the shape [L,B,H,M].

  Returns:
    FAVOR normalizer in noncausal attention.
  """
  all_ones = tf.ones([ks.shape[0]])
  ks_sum = tf.einsum("lbhm,l->bhm", ks, all_ones)
  return tf.einsum("lbhm,bhm->lbh", qs, ks_sum)


@tf.custom_gradient
def causal_numerator(qs, ks, vs):
  """Computes not-normalized FAVOR causal attention A_{masked}V.

  Args:
    qs: query_prime tensor of the shape [L,B,H,M].
    ks: key_prime tensor of the shape [L,B,H,M].
    vs: value tensor of the shape [L,B,H,D].

  Returns:
    Not-normalized FAVOR causal attention A_{masked}V.
  """

  result = []
  sums = tf.zeros_like(tf.einsum("ijk,ijl->ijkl", ks[0], vs[0]))

  for index in range(qs.shape[0]):
    sums = sums + tf.einsum("ijk,ijl->ijkl", ks[index], vs[index])
    result.append(tf.einsum("ijkl,ijk->ijl", sums, qs[index])[None, Ellipsis])

  result = tf.concat(result, axis=0)

  def grad(res_grad):

    grads = tf.zeros_like(tf.einsum("ijk,ijl->ijkl", ks[0], vs[0]))

    gr_sums = sums

    q_grads = []
    k_grads = []
    v_grads = []

    for index in range(qs.shape[0] - 1, -1, -1):

      q_grads.append(
          tf.einsum("ijkl,ijl->ijk", gr_sums, res_grad[index])[None, Ellipsis])
      grads = grads + tf.einsum("ijk,ijl->ijkl", qs[index], res_grad[index])
      k_grads.append(tf.einsum("ijkl,ijl->ijk", grads, vs[index])[None, Ellipsis])
      v_grads.append(tf.einsum("ijkl,ijk->ijl", grads, ks[index])[None, Ellipsis])
      gr_sums = gr_sums - tf.einsum("ijk,ijl->ijkl", ks[index], vs[index])

    q_grads = tf.concat(q_grads[::-1], axis=0)
    k_grads = tf.concat(k_grads[::-1], axis=0)
    v_grads = tf.concat(v_grads[::-1], axis=0)

    return q_grads, k_grads, v_grads

  return result, grad


@tf.custom_gradient
def causal_denominator(qs, ks):
  """Computes FAVOR normalizer in causal attention.

  Args:
    qs: query_prime tensor of the shape [L,B,H,M].
    ks: key_prime tensor of the shape [L,B,H,M].

  Returns:
    FAVOR normalizer in causal attention.
  """

  result = []
  sums = tf.zeros_like(ks[0])

  for index in range(qs.shape[0]):
    sums = sums + ks[index]
    result.append(tf.reduce_sum(qs[index] * sums, axis=2)[None, Ellipsis])

  result = tf.concat(result, axis=0)

  def grad(res_grad):

    k_grad = tf.zeros_like(ks[0])

    gr_sums = sums

    q_grads = []
    k_grads = []

    for index in range(qs.shape[0] - 1, -1, -1):

      q_grads.append(
          tf.einsum("ijk,ij->ijk", gr_sums, res_grad[index])[None, Ellipsis])
      k_grad = k_grad + tf.einsum("ijk,ij->ijk", qs[index], res_grad[index])
      k_grads.append(k_grad[None, Ellipsis])
      gr_sums = gr_sums - ks[index]

    q_grads = tf.concat(q_grads[::-1], axis=0)
    k_grads = tf.concat(k_grads[::-1], axis=0)

    return q_grads, k_grads

  return result, grad


def favor_attention(query,
                    key,
                    value,
                    kernel_transformation,
                    causal,
                    projection_matrix):
  """Computes FAVOR normalized attention.

  Args:
    query: query tensor.
    key: key tensor.
    value: value tensor.
    kernel_transformation: transformation used to get finite kernel features.
    causal: whether attention is causal or not.
    projection_matrix: projection matrix to be used.

  Returns:
    FAVOR normalized attention.
  """
  query_prime = kernel_transformation(query, True,
                                      projection_matrix)  # [B,L,H,M]
  key_prime = kernel_transformation(key, False, projection_matrix)  # [B,L,H,M]
  query_prime = tf.transpose(query_prime, [1, 0, 2, 3])  # [L,B,H,M]
  key_prime = tf.transpose(key_prime, [1, 0, 2, 3])  # [L,B,H,M]
  value = tf.transpose(value, [1, 0, 2, 3])  # [L,B,H,D]

  if causal:
    av_attention = causal_numerator(query_prime, key_prime, value)
    attention_normalizer = causal_denominator(query_prime, key_prime)
  else:
    av_attention = noncausal_numerator(query_prime, key_prime, value)
    attention_normalizer = noncausal_denominator(query_prime, key_prime)
  # TODO(kchoro): Add more comments.
  av_attention = tf.transpose(av_attention, [1, 0, 2, 3])
  attention_normalizer = tf.transpose(attention_normalizer, [1, 0, 2])
  attention_normalizer = tf.expand_dims(attention_normalizer,
                                        len(attention_normalizer.shape))
  return av_attention / attention_normalizer

#Add positional encodings.
'''
This is from rotoformer paper
'''
def rotate_every_two(x):
    x = rearrange(x, '... (d j) -> ... d j', j = 2)
    x1, x2 = tf.unstack(x, axis = -1)
    x = tf.stack([-x2, x1], axis = -1)
    return rearrange(x, '... d j -> ... (d j)')

def apply_rotary_pos_emb(q, k, sinu_pos):
    sinu_pos = rearrange(sinu_pos, '() n (j d) -> n j d', j = 2)
    sin, cos = tf.unstack(sinu_pos, axis = -2)
    sin, cos = map(lambda t: repeat(t, 'b n -> b (n j)', j = 2), (sin, cos))
    q, k = map(lambda t: (t * cos) + (rotate_every_two(t) * sin), (q, k))
    return q, k

class FixedPositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, dim, max_seq_len):
      super().__init__()
      self.dim = dim
      self.max_seq_len = max_seq_len

    def build(self, input_shape):
        self.inv_freq = 1. / (10000 ** (tf.range(start=0, limit=self.dim, delta=2, dtype='float32') / self.dim))
        self.position = tf.range(start=0, limit=self.max_seq_len, delta=1, dtype='float32')
        self.sinusoid_inp = tf.einsum("i,j->ij", self.position, self.inv_freq)
        self.emb = tf.concat((tf.math.sin(self.sinusoid_inp), tf.math.cos(self.sinusoid_inp)), axis=-1)

    def call(self, x):
        return self.emb[None, :x.shape[1], :]

class Attention(tf.keras.layers.Layer):
  """Multi-headed attention layer."""

  def __init__(self,
               hidden_size,
               num_heads,
               attention_dropout,
               max_seq_length,
               kernel_transformation=softmax_kernel_transformation,
               numerical_stabilizer=0.001,
               causal=False,
              # projection_matrix_type=None,
               nb_random_features=16,
               use_rot_emb = False,
               use_spe = False,
               use_mask_pos = False,
               eps = 1e-6,
               normalize = False
               ):
    """Initialize Attention.
    Args:
      hidden_size: int, output dim of hidden layer.
      num_heads: int, number of heads to repeat the same attention structure.
      attention_dropout: float, dropout rate inside attention for training.
      kernel_transformation: transformation used to produce kernel features for
        attention.
      numerical_stabilizer: used to bound away from zero kernel values.
      causal: whether attention is causal or not.
      projection_matrix_type: None if Identity should be used, otherwise random
        projection matrix will be applied.
      nb_random_features: number of random features to be used (relevant only if
        projection_matrix is not None).
    """
    if hidden_size % num_heads:
      raise ValueError(
          "Hidden size ({}) must be divisible by the number of heads ({})."
          .format(hidden_size, num_heads))

    super(Attention, self).__init__()
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.attention_dropout = attention_dropout
    self.kernel_transformation = kernel_transformation
    self.numerical_stabilizer = numerical_stabilizer
    self.causal = causal
 #   self.projection_matrix_type = projection_matrix_type
    self.nb_random_features = nb_random_features
    self.max_seq_length = max_seq_length
    self.use_rot_emb = use_rot_emb
    self.use_spe = use_spe
    self.use_mask_pos = use_mask_pos
    self.eps = eps
    self.normalize = normalize

## Removed projection matrix type since the call is throwing issues

  def build(self, input_shape):
    """Builds the layer."""
    # Layers for linearly projecting the queries, keys, and values.
    size_per_head = self.hidden_size // self.num_heads

    def _glorot_initializer(fan_in, fan_out):
      limit = math.sqrt(6.0 / (fan_in + fan_out))
      return tf.keras.initializers.RandomUniform(minval=-limit, maxval=limit)

    attention_initializer = _glorot_initializer(input_shape.as_list()[-1],
                                                self.hidden_size)
    self.query_dense_layer = util.DenseEinsum(
        output_shape=(self.num_heads, size_per_head),
        kernel_initializer=attention_initializer,
        use_bias=False,
        name="query")
    self.key_dense_layer = util.DenseEinsum(
        output_shape=(self.num_heads, size_per_head),
        kernel_initializer=attention_initializer,
        use_bias=False,
        name="key")
    self.value_dense_layer = util.DenseEinsum(
        output_shape=(self.num_heads, size_per_head),
        kernel_initializer=attention_initializer,
        use_bias=False,
        name="value")

    output_initializer = _glorot_initializer(self.hidden_size, self.hidden_size)
    self.output_dense_layer = util.DenseEinsum(
        output_shape=self.hidden_size,
        num_summed_dimensions=2,
        kernel_initializer=output_initializer,
        use_bias=False,
        name="output_transform")

    if self.use_spe:
       self.spe = SPEFilter(gated=True, code_shape=(self.num_heads, size_per_head))

    super(Attention, self).build(input_shape)

  def get_config(self):
    return {
        "hidden_size": self.hidden_size,
        "num_heads": self.num_heads,
        "attention_dropout": self.attention_dropout,
 #TODO: add in the rest of the configs
    }

  def call(self,
           query_input,
           source_input,
           rpe,
        #   bias,       # remove bias as it will throw error in the call of enformer
           training,
           cache=None,
           decode_loop_step=None):
    """Apply attention mechanism to query_input and source_input.
    Args:
      query_input: A tensor with shape [batch_size, length_query, hidden_size].
      source_input: A tensor with shape [batch_size, length_source,
        hidden_size].
      bias: A tensor with shape [batch_size, 1, length_query, length_source],
        the attention bias that will be added to the result of the dot product.
      training: A bool, whether in training mode or not.
      cache: (Used during prediction) A dictionary with tensors containing
        results of previous attentions. The dictionary must have the items:
            {"k": tensor with shape [batch_size, i, heads, dim_per_head],
             "v": tensor with shape [batch_size, i, heads, dim_per_head]} where
               i is the current decoded length for non-padded decode, or max
               sequence length for padded decode.
      decode_loop_step: An integer, step number of the decoding loop. Used only
        for autoregressive inference on TPU.
    Returns:
      Attention layer output with shape [batch_size, length_query, hidden_size]
    """
    # Linearly project the query, key and value using different learned
    # projections. Splitting heads is automatically done during the linear
    # projections --> [batch_size, length, num_heads, dim_per_head].
    b, n, _ = query_input.shape
    h = self.num_heads

    q = self.query_dense_layer(query_input)
    k = self.key_dense_layer(source_input)
    v = self.value_dense_layer(source_input)

    # if self.projection_matrix_type is None:
    #   projection_matrix = None                  #Had to remove this line.
    # else:
    dim = q.shape[-1]
    tgt_len = k.shape[1]

    if self.use_rot_emb is True: 
      q, k, v = map(lambda t: rearrange(t, 'b n h d -> b h n d', h = h), (q, k, v))

    if self.use_spe is True:
      q, k = self.spe(q, k, rpe)
      v = rearrange(v, 'b n h d -> b h n d', h = h)
      q, k = map(lambda t: rearrange(t, 'b h n d -> b n h d'), (q, k))

    seed = tf.math.ceil(tf.math.abs(tf.math.reduce_sum(q) * BIG_CONSTANT))
    seed = tf.dtypes.cast(seed, tf.int32)
    projection_matrix = create_projection_matrix(
        self.nb_random_features, dim, seed=seed)

    if self.use_mask_pos is True:
      create_kernel = partial(softmax_kernel_transformation, projection_matrix = projection_matrix)
      q, k = map(lambda t: rearrange(t, 'b n h d -> b h n d', h = h), (q, k))
      if self.normalize: 
        q = tf.math.l2_normalize(q,axis=-1)
        k = tf.math.l2_normalize(k,axis=-1)
      q = create_kernel(q, is_query = True)
      k = create_kernel(k, is_query = False)
 
            # Compute the KV matrix
      k = rearrange(k, 'b h n d -> b h d n', h = h) #(batch, head, dim_head, seq_len) ([1, 8, 1000, 16])
     # v = rearrange(v,'b n (h d) -> b n h d', h = h) #(batch, seq_len, head, dim_head)
      q = rearrange(q, 'b h n d -> b n h d', h=h)
      kv = tf.einsum("nhdl,nlhm->nhmdl", k, v)
        
        # Efficient matrix multiplication
      u = tf.signal.rfft(rpe)             #rpe.shape = [num_heads, 2*tgt_len]
        
      y = tf.signal.rfft(kv, fft_length=[2*tgt_len]) #KV.shape  = [bsz, num_heads, v_dim, k_dim, tgt_len]            
      y = tf.einsum("hl,nhmdl->nhmdl", u, y)
      weighted_kv = tf.signal.irfft(y)[:, :,:,:,tgt_len:]

      y1= tf.signal.rfft(k, fft_length=[2*tgt_len]) #k.shape  = [bsz, num_heads, k_dim, tgt_len]
      y1 = tf.einsum("hl,nhdl->nhdl", u, y1)
      weighted_k = tf.signal.irfft(y1)[:, :,:,tgt_len:]
    
        # Compute the normalizer
      Z = 1/(tf.einsum("nlhd,nhdl->nlh", q, weighted_k) + self.eps)
      Z = rearrange(Z, 'n l h -> n h l') #transpose by keeping the batch dim fixed
    
        # Finally compute and return the new values
        # Equivalent to V = torch.einsum("nlhd,nhmdl,nhl->nlhm", Q, weighted_KV, Z)
      attention_output = tf.einsum("nlhd,nhmdl,nhl->nlhm", q, weighted_kv, Z)
 #     attention_output = rearrange(attention_output, 'b n h d -> b n (h d)') #this step differs from the pytorch version
            

# Cache does not work with the spe.
    if cache is not None:
      # Combine cached keys and values with new keys and values.
      if decode_loop_step is not None:
        cache_k_shape = cache["k"].shape.as_list()
        indices = tf.reshape(
            tf.one_hot(decode_loop_step, cache_k_shape[1], dtype=key.dtype),
            [1, cache_k_shape[1], 1, 1])
        k = cache["k"] + k * indices
        cache_v_shape = cache["v"].shape.as_list()
        indices = tf.reshape(
            tf.one_hot(decode_loop_step, cache_v_shape[1], dtype=value.dtype),
            [1, cache_v_shape[1], 1, 1])
        v = cache["v"] + v * indices
      else:
        k = tf.concat([tf.cast(cache["k"], key.dtype), k], axis=1)
        v = tf.concat([tf.cast(cache["v"], value.dtype), v], axis=1)

      # Update cache
      cache["k"] = k
      cache["v"] = v

    if self.use_mask_pos is False:
      attention_output = favor_attention(q, k, v,
                                       self.kernel_transformation, self.causal,
                                       projection_matrix) # shape = [b n h d]
    #  attention_output = rearrange(attention_output, 'b h n d -> b n (h d)')

    attention_output = self.output_dense_layer(attention_output)
    return attention_output


class SelfAttention(Attention):
  """Multiheaded self-attention layer."""

  def call(self,
           query_input,
           rpe,
         #  bias,
           training,
           cache=None,
           decode_loop_step=None):
    return super(SelfAttention, self).call(query_input, query_input, rpe,
                                           training, cache, decode_loop_step)

      # Removed bias in the call of super. 

class PerformerBlock(tf.keras.layers.Layer):
    '''
    This is the performer SELF ATTENTION block.
    '''
    def __init__(self, attention, d_model, dropout=0.1,
                 activation="relu"):
        super(PerformerBlock, self).__init__()


        d_ff = 2*d_model #used in enformer, generally 4*d_model
        self.attention = attention
        self.linear1 = tf.keras.layers.Dense(d_ff)
        self.linear2 = tf.keras.layers.Dense(d_model)
        self.norm1 = tf.keras.layers.LayerNormalization(axis=-1, epsilon=0.001, center=True, scale=True,
                        beta_initializer='zeros', gamma_initializer='ones')
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.activation = getattr(tf.keras.activations, activation)

    def call(self, x, rpe=None, **kwargs):
        """Apply the transformer encoder to the input x.
        Arguments
        ---------
            x: The input features of shape (N, L, E) where N is the batch size,
               L is the sequence length (padded) and E is d_model passed in the
               constructor.
       
        """
        # Normalize the masks
        N = x.shape[0]
        L = x.shape[1]

        # Run self attention and add it to the input
        x = x + self.dropout(self.attention(
            x, x,
            rpe=rpe, 
            **kwargs))
        

        # Run the fully connected part of the layer
        y = self.norm1(x)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))

        return self.norm2(x+y)

class PerformerEncoder(tf.keras.layers.Layer):
    def __init__(self, 
        num_layers, # Number of layers in the encoder
        n_heads, 
        dim, 
        d_model, 
        max_seq_length,
        nb_random_features,
        attention_dropout = .1,
        num_realizations=1,
        norm_layer=None, 
        rel_pos_bins=None, 
        use_spe=False, 
        spe_type=None, 
        kernel_size=None, 
        use_rot_emb = False,
        use_mask_pos = False, 
        normalize = False #normalize keys/queries
        ):

        super(PerformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.norm = norm_layer
        self.dim = dim #dim per head
        self.n_heads = n_heads
        self.d_model = d_model #d_ model = dim per head * n_heads
        self.kernel_size = kernel_size
        self.nb_random_features = nb_random_features
        self.attention_dropout = attention_dropout
        self.num_realizations = num_realizations
        self.max_seq_length = max_seq_length
        self.rel_pos_bins = rel_pos_bins #num_heads * dim
        self.use_rot_emb = use_rot_emb #use rotary positional embeddings in Rotoformer
        self.use_spe = use_spe #gated mechanism for positional embeddings using conv or sine 
        self.spe_type = spe_type #conv/sine spe
        self.use_mask_pos = use_mask_pos #fft masking via Toeplitz matrices
        self.normalize = normalize

        if self.use_mask_pos is True: 
          self.relative_positional_bias = tf.Variable(tf.random.uniform((self.n_heads, 2 * self.rel_pos_bins - 1)))
    

        self.layers = [PerformerBlock(Attention(hidden_size=self.d_model,
          num_heads = self.n_heads,
          nb_random_features = self.nb_random_features,
       # feature_redraw_interval = 1000,
       # generalized_attention = False,
          attention_dropout = self.attention_dropout,
          use_rot_emb = self.use_rot_emb,
          use_spe = self.use_spe,
          use_mask_pos = self.use_mask_pos,
          max_seq_length = self.max_seq_length,
          normalize = self.normalize
      ), d_model = self.d_model) for i in range(self.num_layers)] 

        if self.spe_type== 'sine':
          self.sine_spe = SineSPE(num_heads=self.n_heads, in_features=self.dim, num_sines=self.d_model, num_realizations=self.num_realizations)
        if self.spe_type == 'conv':
          self.conv_spe = ConvSPE(self.n_heads, self.dim, self.d_model, self.kernel_size)

        if self.use_rot_emb is True: 
            self.pos_emb = FixedPositionalEmbedding(self.d_model, self.max_seq_length)
            self.layer_pos_emb = FixedPositionalEmbedding(self.dim, self.max_seq_length)       
    #TODO: DEF BUILD IS NOT WORKING 
    
    def call(self, x, rpe=None, **kwargs):
        """Apply all transformer encoder layers to the input x.
        Arguments
        ---------
            x: The input features of shape (N, L, E) where N is the batch size,
               L is the sequence length (padded) and E is d_model passed in the
               constructor of each transformer encoder layer.
            attn_mask: not compute attention for [PAD] tokens. #TODO: add this to the transformer encoder
        """
        # Normalize the masks
        N = x.shape[0]
        L = x.shape[1]
    
        # We assume that the sequences have the right length and nothing is padded. 
        #TODO: ADD in attention mask if there is a PAD token 

        if self.use_mask_pos is True:
    
            if L <= self.rel_pos_bins:
                rpe = tf.concat((tf.expand_dims(self.relative_positional_bias[:,0], axis=1), 
                                self.relative_positional_bias[:,self.rel_pos_bins-L: self.rel_pos_bins+L-1]), axis=1)
            else:
                rpe = tf.concat([tf.repeat(tf.expand_dims(self.relative_positional_bias[:,0], axis=1), repeats= L-self.rel_pos_bins+1, axis=1), 
                            self.relative_positional_bias,
                            tf.repeat(tf.expand_dims(self.relative_positional_bias[:,-1], axis=1), repeats=L-self.rel_pos_bins, axis=1)], axis=1)

        if self.use_spe is True:   
            if self.spe_type == 'sine':
                rpe = self.sine_spe((self.n_heads, self.max_seq_length))
            elif self.spe_type == 'conv':  #conv gives poor results
                rpe = self.conv_spe(self.n_heads, self.dim)
            else:
                raise ValueError('spe_type not supported')

        if self.use_rot_emb is True:
            x += self.pos_emb(x)
            rpe = self.layer_pos_emb(x)

       

        # Apply all the transformers
        for layer in self.layers:
            x = layer(x, rpe=rpe)

        # Apply the normalization if needed
        if self.norm is not None:
            x = self.norm(x)

        return x
