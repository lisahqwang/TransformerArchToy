
import torch
import torch.nn as nn
import math as m 
import torch.nn.functional as F

from mha import Mha
from residuallayernorm import ResidualLayerNorm
from pwffn import PWFFN
from positional_embedding import Embeddings, PositionalEncoding


###Embedding layer

class Embeddings(nn.Module):
    def __init__(self, vocab_size, padding_idx, d_model):
        super().__init__()
        self.d_model = d_model 
        self.embed = nn.Embedding(vocab_size,  d_model, padding_idx = padding_idx)

    def forward(self, x):
        embedding = self.embed(x)

        return embedding * m.sqrt(self.d_model)
    
    
###Positional Encoding Layer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout = 0.2, max_seq_len = 200, device = 'cpu'):
        #testing drop out to be 0.2, one in five inputs is eliminated every update
        super().__init__()
        self.d_model = d_model 
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, d_model).to(device)
        pos = torch.arange(0, max_seq_len).unsqueeze(1).float()

        two_i = torch.arange(0, d_model, step=2).float()
        denominator = torch.pow(10000, two_i)


        pe[:, 0::2] = torch.sin(pos/ denominator)
        pe[:, 1::2] = torch.cos(pos/ denominator)

        pe = pe.unsqueeze(0)

        # assigns the first argument to a class variable
        # i.e. self.pe
        self.register_buffer("pe", pe)
        

###Multi-head Attention Layer

# d_model (int) is the embedding size

class Mha(nn.Module):
    def __init__(self, num_heads = 2, d_model = 4, dropout = 0.2):
        super().__init__()

        self.d_model = d_model 
        self.num_heads = num_heads

        self.d = d_model // num_heads 

        self.dropout = nn.Dropout(dropout)

        self.linear_Qs = nn.ModuleList([nn.Linear(d_model, self.d)
        for _ in range(num_heads)])
        

        self.linear_Ks = nn.ModuleList([nn.Linear(d_model, self.d)
        for _ in range(num_heads)])

        self.linear_Vs = nn.ModuleList([ nn.Linear(d_model, self.d)
        for _ in range(num_heads)])

        self.mha_linear = nn.Linear(d_model, d_model)


    
    def scaled_dot_product_attention(self, Q, K, V, mask = None):
        # shape Q , K, V: # of batch * [ seq * self.d (d_model / num_heads))    ]
        # q matmul k => batch * [ seq_len * seq_len ] (attention_wieght)
        # output -- batch * [ seq * self.d]

        Q_K_matmul = torch.matmul(Q, K.permute(0,2,1))

        scores = Q_K_matmul / m.sqrt(self.d)

        if mask is not None:
            scores = scores.masked_fill(mask ==0 , -1)

        attention_weight = F.softmax(scores, dim = -1)

        output = torch.matmul(attention_weight, V)

        return output, attention_weight 

    
    def forward(self,q,k,v):
        # shape x = batch * [ seq_len * d_model  ]

        Q = [linear_Q(q)   for linear_Q in self.linear_Qs]
        K = [linear_K(k)  for linear_K in self.linear_Ks]
        V = [linear_V(v)  for linear_V in self.linear_Vs]

        output_per_head = []
        attn_weights_per_head = []

        for Q_, K_, V_ in zip(Q,K,V):
            output, attn_weight = self.scaled_dot_product_attention(Q_, K_, V_)
            output_per_head.append(output)
            attn_weights_per_head.append(attn_weight)

        output = torch.cat( 
            output_per_head, -1
        )
       # num_head [ ]
        attn_weights = torch.stack(attn_weights_per_head).permute(1,0,2,3)

        projection = self.dropout(self.mha_linear(output))

        return projection, attn_weight

### Residual Layer Normalization
class ResidualLayerNorm(nn.Module):
    def __init__(self, d_model, dropout= 0.2):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, residual):
        ln = self.layer_norm(self.dropout(x) + residual )

        return ln

### Feed Forward Layer
class PWFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout = 0.2):
        super().__init__()

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        ff = self.ff(x)

        return  ff

### Encoder Layer
class Encoderlayer(nn.Module):
    def __init__(self, d_ff, d_model, num_heads, dropout = 0.2): 
        super().__init__()


        self.norm_1 = ResidualLayerNorm(d_model, dropout)
        self.norm_2 = ResidualLayerNorm(d_model, dropout)

        self.mha = Mha(d_model, num_heads, dropout)

        self.ff = PWFFN(d_model, d_ff, dropout)



    def forward(self, x):
        # shape(x) = [batch seq_len d_model]

        mha, encoder_attn_weights = self.mha(x)
        norm1 = self.norm_1(mha, x)

        ff = self.ff(norm1)
        norm2 = self.norm_2(ff, norm1)

        return norm2, encoder_attn_weights


### Decoder layer

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.2):
        super().__init__()

        self.norm_1 = ResidualLayerNorm(d_model)
        self.norm_2 = ResidualLayerNorm(d_model)
        self.norm_3 = ResidualLayerNorm(d_model)

        
        self.masked_mha = Mha(d_model, num_heads, dropout)
        self.enc_dec_mha = Mha(d_model, num_heads, dropout)
        
        self.ff = PWFFN(d_model, d_ff)

    def forward(self, x, encoder_outputs, trg_mask, src_mask):

        masked_mha , mask_attn_weights = self.masked_mha(x,x,x, mask = trg_mask)

        norm1 = self.norm_1(masked_mha, x)

        enc_dec_mha , enc_dec_attn_weights =  self.enc_dec_mha(norm1, encoder_outputs, encoder_outputs, mask=src_mask)

        norm2 = self.norm_2(enc_dec_mha, norm1)
        # shape(norm2) = [B x TRG_seq_len x D]

        ff = self.ff(norm2)
        norm3 = self.norm_3(ff, norm2)
        # shape(ff) = [B x TRG_seq_len x D]
        # shape(norm3) = [B x TRG_seq_len x D]

        return norm3, mask_attn_weights, enc_dec_attn_weights
