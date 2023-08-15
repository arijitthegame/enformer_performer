import math
import numpy as np
import torch
import torch.nn.functional as F
# from spe_tf import *
from einops import rearrange, repeat
from functools import partial
from distutils.version import LooseVersion

TORCH_GE_1_8_0 = LooseVersion(torch.__version__) >= LooseVersion('1.8.0')

#import src.layers.util as util
BIG_CONSTANT = 1e8

def orthogonal_matrix_chunk(cols, device = None):
    unstructured_block = torch.randn((cols, cols), device = device)
    if TORCH_GE_1_8_0:
        q, r = torch.linalg.qr(unstructured_block.cpu(), mode = 'reduced')
    else:
        q, r = torch.qr(unstructured_block.cpu(), some = True)
    q, r = map(lambda t: t.to(device), (q, r))
    return q.t()

#TODO 
def create_products_of_givens_rotations(dim, seed, device = None):
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
    q = torch.eye(dim, dim).to(device)
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


def gaussian_orthogonal_random_matrix(m, d, scaling = 0, device = None, struct_mode=False):
    nb_full_blocks = int(m / d)

    block_list = []

    for _ in range(nb_full_blocks):
        if struct_mode:
            q = create_products_of_givens_rotations(d, seed)
        else :
            q = orthogonal_matrix_chunk(d, device = device)

        block_list.append(q)

    remaining_rows = m - nb_full_blocks * d
    if remaining_rows > 0:
        if struct_mode : 
            q = create_products_of_givens_rotations(d, seed) 
        else :
            q = orthogonal_matrix_chunk(nb_columns, device = device)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device = device).norm(dim = 1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,), device = device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    return torch.diag(multiplier) @ final_matrix

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
    #if projection_matrix is None:
    #    return tf.nn.relu(data) + numerical_stabilizer
    #else:
    ratio = 1.0 / torch.sqrt(projection_matrix.shape[0])
    data_dash = ratio * torch.einsum("blhd,md->blhm", data, projection_matrix)
    return F.relu(data_dash, inplace=True) + numerical_stabilizer #check inplace is a valid operation, otherwise make it False


def relu_kernel_transformation_q(data,
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
    #if projection_matrix is None:
    #    return tf.nn.relu(data) + numerical_stabilizer
    #else:
    ratio = 1.0 / torch.sqrt(projection_matrix.shape[0])
    data_dash = ratio * torch.einsum("blhd,md->blhm", data, projection_matrix)
    return F.relu(data_dash)**4 + numerical_stabilizer

def softmax_kernel_transformation(data, is_query, projection_matrix,  normalize_data=True, eps=0.000001, device = None):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    ratio = (projection_matrix.shape[0] ** -0.5)

    projection = repeat(projection_matrix, 'j d -> b h j d', b = b, h = h)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    diag_data = data ** 2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    diag_data = diag_data.unsqueeze(dim=-1)

    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data -
                    torch.amax(data_dash, dim=-1, keepdim=True).detach()) + eps)
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.amax(data_dash, dim=(-1, -2), keepdim=True).detach()) + eps)

    return data_dash.type_as(data)


def noncausal_numerator(qs, ks, vs):
    """Computes not-normalized FAVOR noncausal attention AV.
  Args:
    qs: query_prime tensor of the shape [L,B,H,M].
    ks: key_prime tensor of the shape [L,B,H,M].
    vs: value tensor of the shape [L,B,H,D].
  Returns:
    Not-normalized FAVOR noncausal attention AV.
    """
    kvs = torch.einsum("lbhm,lbhd->bhmd", ks, vs)
    return torch.einsum("lbhm,bhmd->lbhd", qs, kvs)


def noncausal_denominator(qs, ks):
    """Computes FAVOR normalizer in noncausal attention.
  Args:
    qs: query_prime tensor of the shape [L,B,H,M].
    ks: key_prime tensor of the shape [L,B,H,M].
  Returns:
    FAVOR normalizer in noncausal attention.
    """
    all_ones = torch.ones([ks.shape[0]]).float()
    ks_sum = torch.einsum("lbhm,l->bhm", ks, all_ones)
    return torch.einsum("lbhm,bhm->lbh", qs, ks_sum)


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
    #print("qprime", query_prime)
    key_prime = kernel_transformation(key, False, projection_matrix)  # [B,L,H,M]
    #print("kprime", key_prime)
    query_prime = torch.permute(query_prime, (1, 0, 2, 3))  # [L,B,H,M]
    key_prime = torch.permute(key_prime, (1, 0, 2, 3))  # [L,B,H,M]
    value = torch.permute(value, (1, 0, 2, 3))  # [L,B,H,D]
    
    if causal:
        raise NotImplementedError('Casual attention is not yet implemented')
    else:
        av_attention = noncausal_numerator(query_prime, key_prime, value)
        attention_normalizer = noncausal_denominator(query_prime, key_prime)
        
    av_attention = torch.permute(av_attention, (1, 0, 2, 3))
    attention_normalizer = torch.permute(attention_normalizer, (1, 0, 2))
    
    attention_normalizer = torch.unsqueeze(attention_normalizer,
                                        len(attention_normalizer.shape))
    return av_attention / attention_normalizer, key_prime, query_prime


def rotate_every_two(x):
    x = rearrange(x, '... (d j) -> ... d j', j = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d j -> ... (d j)')

def apply_rotary_pos_emb(q, k, sinu_pos):
    sinu_pos = sinu_pos.type(q.dtype)
    sinu_pos = rearrange(sinu_pos, '() n (j d) -> n j d', j = 2)
    sin, cos = sinu_pos.unbind(dim = -2)
    sin, cos = map(lambda t: repeat(t, 'b n -> b (n j)', j = 2), (sin, cos))
    q, k = map(lambda t: (t * cos) + (rotate_every_two(t) * sin), (q, k))
    return q, k 

class Attention(nn.Module):
    """Multi-headed attention layer."""

    def __init__(self,
                 hidden_size,
                 num_heads,
                 kernel_transformation=softmax_kernel_transformation,
                 numerical_stabilizer=0.001,
                causal=False,
                   nb_random_features=16,
                   use_rot_emb = True,
                   use_mask_pos = False,
                   eps = 1e-6,
                   normalize = True,
                   seed=42,
                 q_init=None,
                 k_init=None,
                 v_init=None,
                 att_output=None,
                 load_init = False
                   ):
        
#     """Initialize Attention.
    
#     Args:
#         hidden_size: int, output dim of hidden layer.
#         num_heads: int, number of heads to repeat the same attention structure.
#         attention_dropout: float, dropout rate inside attention for training.
#         kernel_transformation: transformation used to produce kernel features for
#             attention.
#         numerical_stabilizer: used to bound away from zero kernel values.
#         causal: whether attention is causal or not.
#         projection_matrix_type: None if Identity should be used, otherwise random
#             projection matrix will be applied.
#         nb_random_features: number of random features to be used (relevant only if
#             projection_matrix is not None).
            
#     """
        

        if hidden_size % num_heads:
            raise ValueError(
                "Hidden size ({}) must be divisible by the number of heads ({})."
                .format(hidden_size, num_heads))

        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.kernel_transformation = kernel_transformation
        self.numerical_stabilizer = numerical_stabilizer
        self.causal = causal
        self.nb_random_features = nb_random_features
        self.use_rot_emb = use_rot_emb
        self.use_mask_pos = use_mask_pos
        self.eps = eps
        self.normalize = normalize
        self.seed = seed
        self.load_init=load_init
        self.q_init=q_init
        self.k_init=k_init
        self.v_init=v_init
        self.att_output=att_output
        self.scaling = scaling

        self.dim_heads = self.hidden_size//self.num_heads
        
        self.create_projection = partial(gaussian_orthogonal_random_matrix, m = self.nb_random_features, d = self.dim_heads, scaling = self.scaling)
        projection_matrix = self.create_projection()
        self.register_buffer('projection_matrix', projection_matrix)
        self.query_dense_layer = None
        self.key_dense_layer = None
        self.value_dense_layer = None

    def forward(self,
           query_input,
           source_input,
           rpe):
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

        q = tf.cast(self.query_dense_layer(query_input),dtype=tf.float32)
        k = tf.cast(self.key_dense_layer(source_input),dtype=tf.float32)
        v = tf.cast(self.value_dense_layer(source_input),dtype=tf.float32)
        
        if self.kernel_transformation == 'relu_kernel_transformation':
            kernel_transform = relu_kernel_transformation
        elif self.kernel_transformation == 'relu_kernel_transformation_q':
            kernel_transform = relu_kernel_transformation_q
        else:
            kernel_transform = softmax_kernel_transformation

        dim = q.shape[-1]
        tgt_len = k.shape[1]
        
        
        if self.use_mask_pos is True:

            create_kernel = partial(kernel_transform, projection_matrix= self.projection_matrix)
            q, k = map(lambda t: tf.transpose(t, [0,2,1,3]), (q,k))

                       #rearrange(t, 'b n h d -> b h n d', h = h), (q, k))
            if self.normalize: 
                q = tf.math.l2_normalize(q,axis=-1)
                k = tf.math.l2_normalize(k,axis=-1)
            q_prime = create_kernel(q, is_query = True)
            k_prime = create_kernel(k, is_query = False)
            #k_prime = rearrange(k_prime, 'b h n d -> b h d n', h=h) #(batch, head, dim_head, seq_len) ([1, 8, 1000, 16])
            k_prime = tf.transpose(k_prime, [0,1,3,2])
            #q_prime = rearrange(q_prime, 'b h n d -> b n h d', h=h)
            q_prime = tf.transpose(q_prime, [0,2,1,3])

            kv = tf.einsum("nhdl,nlhm->nhmdl", k_prime, v)

            # Efficient matrix multiplication
            u = tf.signal.rfft(tf.cast(rpe,dtype=tf.float32))          #rpe.shape = [num_heads, 2*tgt_len]
            #print("u", u.shape)

            y = tf.signal.rfft(tf.cast(kv, dtype=tf.float32),
                               fft_length=[2*tgt_len]) #KV.shape  = [bsz, num_heads, v_dim, k_dim, tgt_len]  
            y = tf.einsum("hl,nhmdl->nhmdl", u, y)
            weighted_kv = tf.cast(tf.signal.irfft(y)[:, :,:,:,tgt_len:],dtype=tf.float32)

            y1= tf.signal.rfft(tf.cast(k_prime,dtype=tf.float32) ,
                               fft_length=[2*tgt_len]) #k.shape  = [bsz, num_heads, k_dim, tgt_len]

            y1 = tf.einsum("hl,nhdl->nhdl", u, y1)
            weighted_k = tf.cast(tf.signal.irfft(y1)[:, :,:,tgt_len:],dtype=tf.float32)
            #print("weighted k", weighted_k.shape)

            # Compute the normalizer
            Z = 1/(tf.einsum("nlhd,nhdl->nlh", q_prime, weighted_k) + self.eps)
            #Z = rearrange(Z, 'n l h -> n h l') #transpose by keeping the batch dim fixed
            Z = tf.transpose(Z, [0,2,1])
            #print("Z rearrange", Z.shape)

            # Finally compute and return the new values
            # Equivalent to V = torch.einsum("nlhd,nhmdl,nhl->nlhm", Q, weighted_KV, Z)
            attention_output = tf.einsum("nlhd,nhmdl,nhl->nlhm", q_prime, weighted_kv, Z)
            # attention_output = rearrange(attention_output, 'b n h d -> b n (h d)')
            #print("attention_output rearrange", attention_output.shape)

        if self.use_rot_emb is True and self.use_mask_pos is False:
            #q, k, v = map(lambda t: rearrange(t, 'b n h d -> b h n d', h = h), (q, k, v))
            q,k = apply_rotary_pos_emb(q,k,rpe)
            #k = apply_rotary_pos_emb(rpe, k)
            #q, k = apply_rotary_pos_emb(q,k,rpe)
            attention_output, k_prime, q_prime = favor_attention(q, k, v,
                                       kernel_transform, self.causal,
                                       self.projection_matrix)
        if rpe is None and not self.use_rot_emb and not self.use_mask_pos:
            attention_output, k_prime, q_prime = favor_attention(q, k, v,
                                       kernel_transform, self.causal,
                                       self.projection_matrix)
        
        attention_output = self.output_dense_layer(attention_output)
        #print("attn2", attention_output.shape)
        return attention_output, k_prime, q_prime

