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

def softmax_kernel_transformation(data, projection_matrix, is_query, normalize_data=True, eps=0.000001, device = None):
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
    query_prime = tf.transpose(query_prime, [1, 0, 2, 3])  # [L,B,H,M]
    key_prime = tf.transpose(key_prime, [1, 0, 2, 3])  # [L,B,H,M]
    value = tf.transpose(value, [1, 0, 2, 3])  # [L,B,H,D]
    
    if causal:
        raise NotImplementedError('Casual attention is not yet implemented')
    else:
        av_attention = noncausal_numerator(query_prime, key_prime, value)
        attention_normalizer = noncausal_denominator(query_prime, key_prime)
        
  # TODO(kchoro): Add more comments.
    av_attention = tf.transpose(av_attention, [1, 0, 2, 3])
    #print("avattn", av_attention.shape)
    attention_normalizer = tf.transpose(attention_normalizer, [1, 0, 2])
    
    attention_normalizer = tf.expand_dims(attention_normalizer,
                                        len(attention_normalizer.shape))
    return av_attention / attention_normalizer, key_prime, query_prime
