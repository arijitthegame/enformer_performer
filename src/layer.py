import torch
from torch import nn

import util
from performer_attention import Attention

class abs_sin_PE(nn.Module):
    def __init__(self, 
                 input_shape,
                 **kwargs):
        """basic absolute sinusoidal PE layer
        Args:
            positional_dropout_rate: dropout rate for positional embeddings
        """
        super().__init__()
        self.input_shape = input_shape
        self._pe = util.sinusoidal(self.input_shape)

    def forward(self, x):
        return self._pe + x


class rotary_PE(nn.Module):
    def __init__(self, 
                 input_shape,
                 positional_dropout_rate: float, 
                 **kwargs):
        """basic absolute sinusoidal PE layer
        Args:
            positional_dropout_rate: dropout rate for positional embeddings
        """
        super().__init__()
        self.input_shape = input_shape
        self._positional_dropout_rate = positional_dropout_rate
        self._dropout = nn.Dropout(self._positional_dropout_rate)
        self._pe = util.sinusoidal(self.input_shape)

    def forward(self, x):
        return self._dropout(self._pe + x)
    
    
class TargetLengthCrop1D(nn.Module):
    """Crop sequence to match the desired target length."""
    def __init__(self,
               uncropped_length: int = 768,
               target_length: int = 448,
             ):
        super().__init__()
        self._target_length = target_length
        self._uncropped_length = uncropped_length
        
    def forward(self, x):
        if self._target_length is None:
            return x
        trim = (self._uncropped_length - self._target_length) // 2
        if trim < 0:
            raise ValueError('inputs longer than target length')
        elif trim == 0:
            return x
        else:
            return x[..., trim:-trim, :]
        
        
class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        self.inv_freq = 1. / (10000 ** (torch.arange(0, self.dim, 2, dtype=torch.float32)/ self.dim))
        self.position = torch.arange(start=0, end=self.max_seq_len, dtype=torch.float32)
        self.sinusoid_inp = torch.einsum("i,j->ij", self.position, self.inv_freq)
        self.emb = torch.cat((torch.sin(self.sinusoid_inp), 
                              torch.cos(self.sinusoid_inp)), dim=-1)

    def forward(self, x):
        return self.emb[None, :x.shape[1], :]


class FFN(nn.Module):
    def __init__(self, 
                 num_channels: int, 
                 dropout_rate: float,
                 ffn_widening=2
                 **kwargs):
        super().__init__()
        """FFN/MLP layer for transformer block
        Args:
            num_channels: num output channels
            widening: scaling factor for how many channels to start w/
                      e.g. widening = 2, num_channels = 12 means start w/ 24
            dropout_rate: dropout rate used throughout network
            name: Module name.
        """
        self.ffn_channels = num_channels
        self.ffn_widening = ffn_widening
        self.ffn_dropout = dropout_rate
        
        self.FFN_layer_norm = nn.LayerNorm(self.ffn_channels)
        self.FFN_dense_wide = nn.Linear(self.ffn_channels, self.ffn_channels*self.ffn_widening
                                       )
        self.dropout = nn.Dropout(self.ffn_dropout,)
        self.relu = nn.ReLU()
        self.FFN_dense_narrow = nn.Linear(self.ffn_channels*self.ffn_widening, self.ffn_channels, 
                                         )
    
    def forward(self, inputs):
        x = self.FFN_layer_norm(inputs)
        x = self.FFN_dense_wide(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.FFN_dense_narrow(x)
        x = self.dropout(x)
        return x
    

class Performer(nn.Module):
    def __init__(self,
                 input_dim,
                 d_model,
                 normalize,
                 hidden_size: int,
                 num_heads: int,
                 seed: int,
                 dropout_rate: float,
                 numerical_stabilizer: float,
                 nb_random_features: int,
                 max_seq_length: int,
                 rel_pos_bins=None,
                 kernel_transformation: str = 'relu_kernel_transformation',
                 use_mask_pos: bool = False,
                 use_rot_emb: bool = True,
                 LN_gamma_init = None,
                 LN_beta_init= None,
                 q_init=None,
                 k_init=None,
                 v_init=None,
                 att_output=None,
                 FFN_LN_gamma_init=None,
                 FFN_LN_beta_init=None,
                 FFN_kernel1_init=None,
                 FFN_bias1_init=None,
                 FFN_kernel2_init=None,
                 FFN_bias2_init=None,
                 load_init: bool = False,
                 **kwargs):
        super().__init__()
        """Transformer block w/ performer attention
        Args:
            hidden size: ~channel dimension for transformer input
            num_heads: num attention heads
            numerical_stabilizer: small float for stability
            nb_random_features: dim for projection matrix
            widening: scaling factor for how many channels to start w/
                      e.g. widening = 2, num_channels = 12 means start w/ 24
            dropout_rate: transformer MLP dropout rate
            kernel_transformation: softmax or relu kernel transform for fast att.
            positional_encoding_type: absolute sinusoidal or relative(rotary)
            name: Module name.
        """
        self.input_dim=input_dim
        self.hidden_size=hidden_size
        self.num_heads=num_heads
        self.dropout_rate=dropout_rate
        self.kernel_transformation=kernel_transformation 
        self.numerical_stabilizer=numerical_stabilizer
        self.max_seq_length = max_seq_length
        self.nb_random_features=nb_random_features
        self.rel_pos_bins = rel_pos_bins
        self.use_rot_emb=use_rot_emb
        self.use_mask_pos=use_mask_pos
        self.d_model=d_model
        self.normalize=normalize
        self.seed=seed
        self.load_init=load_init
        self.FFN_LN_gamma_init=None,
        self.FFN_LN_beta_init=FFN_LN_beta_init
        self.FFN_kernel1_init=FFN_kernel1_init
        self.FFN_bias1_init=FFN_bias1_init
        self.FFN_kernel2_init=FFN_kernel2_init
        self.FFN_bias2_init=FFN_bias2_init
        
        self.layer_norm = nn.LayerNorm(self.input_dim)
        

        self.self_attention = Attention(hidden_size=self.d_model,
                                               num_heads=self.num_heads,
                                               nb_random_features=self.nb_random_features,
                                               use_rot_emb=self.use_rot_emb,
                                               use_mask_pos=self.use_mask_pos,
                                               normalize=self.normalize,
                                               kernel_transformation=self.kernel_transformation,
                                               numerical_stabilizer=self.numerical_stabilizer,
                                               seed=self.seed,
                                               q_init=q_init,
                                               k_init=k_init,
                                               v_init=v_init,
                                               att_output=att_output,
                                               load_init = self.load_init,
                                               **kwargs)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.FFN = FFN(num_channels=self.hidden_size,
                       dropout_rate=self.dropout_rate,
                       **kwargs)         
    
    
    def forward(self, inputs, rpe=None):
        x = self.layer_norm(inputs)
        x, k_prime, q_prime = self.self_attention(x,x, rpe=rpe)
        x = self.dropout(x)
        mha_output = x + inputs
        ## ffn
        FFN_out = self.FFN(mha_output)
        #return self.layer_norm(FFN_out + mha_output), k_prime, q_prime
        return (FFN_out + mha_output), k_prime, q_prime


class Performer_Encoder(kl.Layer):
    def __init__(self,
                 num_layers,
                 num_heads,
                 dim,
                 d_model,
                 max_seq_length,
                 nb_random_features,
                 hidden_size,
                 numerical_stabilizer,
                 dropout_rate = 0.40,
                 rel_pos_bins=None,
                 use_rot_emb=True,
                 use_mask_pos=False,
                 normalize=True,
                 norm=True,
                 seed=42,
                 load_init=True,
                 inits=None,
                 kernel_transformation: str = 'relu_kernel_transformation',
                 name = 'performer_stack',
                 **kwargs):
        
        
        super().__init__(name=name, **kwargs)
        """Performer Encoder block
        Args:
            hidden size: ~channel dimension for transformer input
            num_heads: num attention heads
            attention_dropout: post attention layer dropout rate
            numerical_stabilizer: small float for stability
            nb_random_features: dim for projection matrix
            widening: scaling factor for how many channels to start w/
                      e.g. widening = 2, num_channels = 12 means start w/ 24
            dropout_rate: transformer MLP dropout rate
            dropout_rate: dropout rate used throughout network
            kernel_transformation: softmax or relu kernel transform for fast att.
            positional_encoding_type: absolute sinusoidal or relative(rotary)
            name: Module name.
        """
        self.num_layers=num_layers
        self.num_heads=num_heads
        self.dim=dim
        self.hidden_size=hidden_size
        self.d_model=d_model
        self.max_seq_length=max_seq_length
        self.nb_random_features=nb_random_features
        self.numerical_stabilizer=numerical_stabilizer
        self.rel_pos_bins=rel_pos_bins#None#rel_pos_bins
        self.use_rot_emb=use_rot_emb
        self.use_mask_pos=use_mask_pos
        self.normalize=normalize
        self.norm=norm
        self.kernel_transformation=kernel_transformation
        self.seed=seed
        self.dropout_rate=dropout_rate
        self.load_init=load_init
        self.inits=inits
        
        self.layers = nn.Modulelist([Performer(d_model=self.d_model, 
                                 normalize=self.normalize,
                                 hidden_size=self.hidden_size,
                                 num_heads=self.num_heads,
                                 dropout_rate=self.dropout_rate,
                                 numerical_stabilizer=self.numerical_stabilizer,
                                 nb_random_features=self.nb_random_features,
                                 max_seq_length=self.max_seq_length,
                                 rel_pos_bins=self.rel_pos_bins,
                                 kernel_transformation=self.kernel_transformation,
                                 use_mask_pos=self.use_mask_pos,
                                 seed=self.seed,
                                 use_rot_emb=self.use_rot_emb,
                                 load_init=self.load_init,
                                 LN_gamma_init = inits["LN_g" + str(i)] if self.load_init else None,
                                 LN_beta_init=  inits["LN_b" + str(i)] if self.load_init else None,
                                 q_init= inits["SA_q" + str(i)] if self.load_init else None,
                                 k_init= inits["SA_k" + str(i)] if self.load_init else None,
                                 v_init= inits["SA_v" + str(i)] if self.load_init else None,
                                 att_output= inits["SA_O" + str(i)] if self.load_init else None,
                                 FFN_LN_gamma_init= inits["FFN_LN_g" + str(i)] if self.load_init else None,
                                 FFN_LN_beta_init= inits["FFN_LN_b" + str(i)] if self.load_init else None,
                                 FFN_kernel1_init= inits["FFN_wide_k" + str(i)] if self.load_init else None,
                                 FFN_bias1_init= inits["FFN_wide_b" + str(i)] if self.load_init else None,
                                 FFN_kernel2_init= inits["FFN_narr_k" + str(i)] if self.load_init else None,
                                 FFN_bias2_init= inits["FFN_narr_b" + str(i)] if self.load_init else None,
                                 **kwargs) for i in range(self.num_layers)]
        )
        
        self.layer_norm = nn.LayerNorm(self.dim)
        
        if self.use_mask_pos:
            self.relative_positional_bias = torch.rand(self.num_heads, 2 * self.rel_pos_bins - 1)       
        if self.use_rot_emb: 
            self.pos_emb = FixedPositionalEmbedding(self.d_model, self.max_seq_length)
            self.layer_pos_emb = FixedPositionalEmbedding(self.dim, self.max_seq_length)       
        
        if self.use_mask_pos:
            if L <= self.rel_pos_bins:
                self.rpe = torch.cat((torch.unsqueeze(self.relative_positional_bias[:,0], 1), 
                            self.relative_positional_bias[:,self.rel_pos_bins-L: self.rel_pos_bins+L-1]), 
                            dim=1)
            else:
                self.rpe = torch.cat((torch.repeat_interleave(torch.unsqueeze(self.relative_positional_bias[:,0], 1), 
                                                             L-self.rel_pos_bins+1, 
                                                             dim=1), 
                        self.relative_positional_bias,
                        torch.repeat_interleave(torch.unsquueze(self.relative_positional_bias[:,-1], 1), L-self.rel_pos_bins, dim=1)),
                          dim=1)
    
    
    def forward(self, x):
        att_matrices={}

        for idx,layer in enumerate(self.layers):
            if self.use_rot_emb is True:
                x += self.pos_emb(x)
                rpe = self.layer_pos_emb(x)
                x,k_prime,q_prime = layer(x, rpe=rpe)
                att_matrices['layer_' + str(idx)] = (k_prime,q_prime)
                
            if self.use_mask_pos is True:
                x,k_prime,q_prime = layer(x, rpe=self.rpe)
                att_matrices['layer_' + str(idx)] = (k_prime,q_prime)
                
        if self.norm:
            x = self.layer_norm(x)
            
        return x,att_matrices
 

