# Author: Xiaoli Wang
# Email: xiaoliw1995@gmail.com
# @Time 2024/3/30
import torch
import torch.nn as nn

import copy
import math

from einops import repeat
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F

'''
This code is adapted from:
https://github.com/justsmart/RecFormer
'''
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def setEmbedingModel(d_list, d_out):
    return nn.ModuleList([nn.Linear(d, d_out) for d in d_list])


class Mlp(nn.Module):
    """ Transformer Feed-Forward Block """

    def __init__(self, in_dim, mlp_dim, out_dim, dropout_rate=0.2):
        super(Mlp, self).__init__()

        # init layers
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, out_dim)
        self.act = nn.GELU()
        if dropout_rate > 0.0:
            self.dropout1 = nn.Dropout(dropout_rate)
            self.dropout2 = nn.Dropout(dropout_rate)
        else:
            self.dropout1 = None
            self.dropout2 = None

    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)

        if self.dropout1:
            out = self.dropout1(out)
        out = self.fc2(out)
        if self.dropout1:
            out = self.dropout2(out)
        return out


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

#q, k, v:  B, heads, view, d_model/heads
def attention(q, k, v, d_k, mask=None, src_mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # scores shape is [bs heads view view]
    # k.transpose(-2, -1): B, heads, d_model/heads, view

    if src_mask is not None:
        if mask is not None:
            mask = mask.unsqueeze(1).float()  # (bs, 1, view)
            mask = mask.unsqueeze(-1).matmul(mask.unsqueeze(-2))  # mask shape is [bs 1 view view]

            src_mask = src_mask.unsqueeze(1).float()  # (bs, 1, view)
            src_mask = src_mask.unsqueeze(-1).matmul(src_mask.unsqueeze(-2))  # mask shape is [bs 1 view view]

            mask_all = mask.matmul(src_mask)
            scores = scores.masked_fill(mask_all == 0, -1e9)  # mask invalid view
        else:
            src_mask = src_mask.unsqueeze(1).float()  # (bs, 1, view)
            src_mask = src_mask.unsqueeze(-1).matmul(src_mask.unsqueeze(-2))  # mask shape is [bs 1 view view]
            scores = scores.masked_fill(src_mask == 0, -1e9)  # mask invalid view

    scores = F.softmax(scores, dim=-1)
    # scores = scores.matmul(src_mask)

    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output

class linear_attention(nn.Module):
    def __init__(self, in_dim):
        super(linear_attention, self).__init__()
        self.layerQ = nn.Linear(in_dim, in_dim)
        self.layerK = nn.Linear(in_dim, in_dim)
        self.layerV = nn.Linear(in_dim, in_dim)
        self.proj_drop = nn.Dropout(p=0.1)
        self.proj = nn.Linear(in_dim, in_dim)
        self.initialize()

    def initialize(self):
        self.layerQ.reset_parameters()
        self.layerK.reset_parameters()
        self.layerV.reset_parameters()

    def forward(self, node_emb, label_emb, tau=0.5, src_mask=None):
        # pdb.set_trace()
        Q = self.layerQ(label_emb)  # [128, 3, 512]
        K = self.layerK(node_emb)
        V = self.layerV(node_emb)   # [128, 3, 512]
        attention_score = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(node_emb.shape[-1]) # [128, 3, 3]

        # NOTE:注意mask的形状
        # if src_mask is not None:
        #     src_mask = src_mask.unsqueeze(1)
        #     # src_mask = torch.matmul(src_mask.float().t().unsqueeze(-1), src_mask.float().t().unsqueeze(1))
        #     src_mask = torch.matmul(src_mask.transpose(-2, -1), src_mask)
        #     attention_score = attention_score.masked_fill(src_mask == 0, -1e9)  # mask invalid view

        mask = torch.tril(torch.ones_like(attention_score))
        attention_score = attention_score.masked_fill(mask == 0, -1e9)

        attention_weight = F.softmax(attention_score * tau, dim=-1)
        z = torch.matmul(self.proj_drop(attention_weight), V)
        z = self.proj(z)

        return z

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, k, v, src_mask=None):

        bs = k.size(1)  # [view, bs, dmodel]

        q = k
        # perform linear operation and split into N heads
        k = self.k_linear(k).view(-1, bs, self.h, self.d_k)  # [3, 128, 4, 64]
        q = self.q_linear(q).view(-1, bs, self.h, self.d_k)
        v = self.v_linear(v).view(-1, bs, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model/h
        k = k.transpose(1, 2)  # [3, 4, 128, 64]
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # [3, 4, 128, 128]

        # NOTE:注意mask的形状
        src_mask = src_mask.float().t().unsqueeze(1)  # [3, 1, 128]
        src_mask = torch.matmul(src_mask.unsqueeze(1), src_mask.unsqueeze(-1))  # [3, 1, 128, 128]
        # src_mask = src_mask.unsqueeze(-2)  # mask shape is [bs 1 1 view]
        scores = scores.masked_fill(src_mask == 0, -1e9)  # mask invalid view
        scores = torch.matmul(scores, v)  # [3, 4, 128, 128] * [3, 4, 128, 64] --> [3, 4, 128, 64]


        scores = F.softmax(scores, dim=-1)  # [3, 4, 128, 64]
        concat = scores.transpose(1, 2).contiguous() \
            .view(-1, bs, self.d_model)  # [3, 128, 4, 64]

        output = self.out(concat)

        return output


class LinearAttention(nn.Module):
    def __init__(self, d_model, key_dim, attn_ratio=4):
        super().__init__()
        self.scale = key_dim ** -0.5
        self.scale = d_model ** -0.5
        self.key_dim = key_dim
        self.d = int(attn_ratio * key_dim)
        self.attn_ratio = attn_ratio

        self.query_proj = nn.Linear(d_model, key_dim)
        self.key_proj = nn.Linear(d_model, key_dim)
        self.value_proj = nn.Linear(d_model, self.d)
        self.proj = nn.Linear(self.d, d_model)

        # self.query_proj = nn.Linear(d_model, d_model)
        # self.key_proj = nn.Linear(d_model, d_model)
        # self.value_proj = nn.Linear(d_model, d_model)
        # self.proj = nn.Linear(d_model, d_model)

    def forward(self, query, key_value):
        B, L, D = query.shape
        q = self.query_proj(query)  # B, L, key_dim
        k = self.key_proj(key_value)  # B, L, key_dim
        v = self.value_proj(key_value)  # B, L, d

        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, L, L
        attn = F.softmax(attn, dim=-1)
        out = attn @ v  # B, L, d
        out = self.proj(out)  # B, L, d_model
        return out


class CascadedAttention(nn.Module):
    def __init__(self, d_model, key_dim, attn_ratio=4, num_modalities=3):
        super().__init__()
        self.layers = nn.ModuleList([LinearAttention(d_model, key_dim, attn_ratio) for _ in range(num_modalities - 1)])
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):  # x1 (B, num_modalities, d_model)
        B, num_modalities, D = x.shape
        query = x[:, 0, :].unsqueeze(1)
        attn_out = []
        for i in range(num_modalities-1):
            if i > 0:
                query = x[:, i, :].unsqueeze(1) + query  # Initial query from modality 0
            key_value = x[:, i+1, :].unsqueeze(1)
            query = self.layers[i](query, key_value)
            attn_out.append(query)
        x = query
        # x1 = self.proj(torch.concatenate(attn_out, dim=1))
        return x

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.2):
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout_1 = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout_1(F.relu(self.linear_1(x)))
        x = self.dropout_2(self.linear_2(x))
        return x


class GLU(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.1):
        super(GLU, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(input_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.dropout(self.linear1(x)) * self.sigmoid(self.linear2(x))


class GLU_FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.1):
        super(GLU_FFN, self).__init__()
        self.glu = GLU(input_dim, hidden_dim, dropout_rate)
        self.linear = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.layernorm = nn.LayerNorm(input_dim)

    def forward(self, x):
        residual = x
        x = self.glu(x)
        x = self.dropout(self.linear(x))
        x = self.layernorm(x + residual)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, view_num, dropout=0.1):
        super().__init__()
        # self.norm_1 = Norm(d_model)
        # self.norm_2 = Norm(d_model)
        self.norm_1 = nn.LayerNorm([view_num, d_model])
        self.norm_2 = nn.LayerNorm([view_num, d_model])
        # self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.label_attentive = linear_attention(d_model)
        self.cascade = CascadedAttention(d_model, 64, 4, view_num)

        self.ff = FeedForward(d_model, dropout=0.2)
        # self.ff = Mlp(in_dim=d_model, mlp_dim=int(d_model), out_dim=d_model, dropout_rate=dropout)
        # self.ff = GLU_FFN(input_dim=d_model, hidden_dim=d_model * 4)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, tau, src_mask):
        x2 = self.norm_1(x)
        # x1 = x1 + self.dropout_1(self.label_attentive(x2, x2, src_mask))   # x2: [B, view, dim]
        # x1 = x1 + self.dropout_1(self.attn(x2, x2, src_mask))
        attn = self.dropout_1(self.label_attentive(x2, x2, tau, src_mask))
        # attn = self.dropout_1(self.cascade(x2))
        x = x + attn    # attention 后输出的形状为[bs, d_model]
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, N, heads, view_num, dropout):
        super().__init__()
        self.N = N

        # self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, view_num, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, src, tau, src_mask):
        x = src
        # x1 = self.embed(src)
        # x1 = self.pe(x1)
        for i in range(self.N):
            x = self.layers[i](x, tau, src_mask)
        return self.norm(x)


class Multi_Encoder(nn.Module):
    def __init__(self, d_list, d_out, hidden_layers=3, hidden_dim=128, dropout_rate=0.1):
        super(Multi_Encoder, self).__init__()
        layers = []

        for d in d_list:
            layer = []
            in_dim = d
            for _ in range(hidden_layers):
                layer.append(nn.Linear(in_dim, hidden_dim))
                layer.append(nn.ReLU())
                layer.append(nn.Dropout(dropout_rate))
                in_dim = hidden_dim
            layer.append(nn.Linear(hidden_dim, d_out))
            layers.append(nn.Sequential(*layer))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        # 假设x是一个列表，包含多个张量，每个张量需要通过一个不同的编码器
        outputs = [layer(x_i) for layer, x_i in zip(self.layers, x)]
        return outputs


class Transformer(nn.Module):
    def __init__(self, d_model, N, heads, view_num, dropout):
        super().__init__()
        self.encoder = Encoder(d_model, N, heads, view_num, dropout)

    def forward(self, src, tau, src_mask):
        e_outputs = self.encoder(src, tau, src_mask)

        return e_outputs


# class Classifier(nn.Module):
#     def __init__(self, input_dim, classes):
#         super(Classifier, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(input_dim, classes)
#         )
#
#     def forward(self, x1):
#         return self.fc(x1)
class Classifier(nn.Module):
    def __init__(self, nhid, nclass, dropout=0., with_bn=True, with_bias=True):
        super(Classifier, self).__init__()
        self.with_bn = with_bn
        self.layer1 = nn.Linear(nhid, int(nhid/2), bias=with_bias)
        self.layer2 = nn.Linear(int(nhid/2), nclass, bias=with_bias)
        if with_bn:
            self.bn1 = nn.BatchNorm1d(int(nhid/2))

        self.dropout = dropout

        self.initialize()

    def initialize(self):
        self.layer1.reset_parameters()
        self.layer2.reset_parameters()
        self.bn1.reset_parameters()

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layer2(x)

        # return F.log_softmax(x1, dim=1)
        return x

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=200, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        pe = Parameter(self.pe[:, :seq_len])
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return self.dropout(x)
class Model(nn.Module):
    def __init__(self, d_list,
                 d_model, n_layers, heads,
                 class_num, tau,
                 dropout,
                 ):

        super().__init__()
        self.class_num = class_num
        self.view_num = len(d_list)
        self.ETrans = Transformer(d_model, n_layers, heads, self.view_num, dropout)
        self.embeddinglayers = setEmbedingModel(d_list, d_model)  # embedding
        self.Classifier = Classifier(d_model, class_num)
        self.linear = nn.Linear(d_model, class_num)
        self.PositionalEncoder = PositionalEncoder(d_model, self.view_num + 1)
        self.MultiEncoder = Multi_Encoder(d_list, d_model, 3, 256, 0.)
        self.attn = linear_attention(d_model)
        self.norm_1 = nn.LayerNorm([self.view_num+1, d_model])
        self.norm_2 = nn.LayerNorm([self.view_num+1, d_model])
        self.dropout_1 = nn.Dropout(dropout)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.to_latent = nn.Identity()
        self.tau = tau

    def forward(self, x, src_mask=None):    # gt: [Bs, n_classes]
        for v in range(self.view_num):  # encode input view to features with same dimension
            x[v] = self.embeddinglayers[v](x[v])
        # x1 = self.MultiEncoder(x1)    # list
        x = torch.stack(x, dim=1)  # B,view,dim
        # x1 = self.PositionalEncoder(x1)
        b, n, _ = x.shape
        # cls_tokens = repeat(self.cls_token, '() n d->b n d', b=b)
        # x1 = torch.cat((cls_tokens, x1), dim=1)

        x = self.ETrans(x, self.tau, src_mask)  # mask/src_mask: [B, view]

        # step2: linear attention
        # x2 = self.norm_1(x1)
        # x1 = x1 + self.dropout_1(self.attn(x2, x2, src_mask))  # x2: [B, view, dim]
        # # x1 = x1 + self.dropout_1(self.attn(x2, x2, src_mask))
        # x1 = self.norm_2(x1)
        # EncX = x1

        # feature fusion
        # x1 = x1.mul(src_mask.unsqueeze(-1))
        # x_new = x1.reshape(b, -1)
        # print(x_new.shape)
        x = torch.einsum('bvd->bd', x)
        wei = 1 / torch.sum(src_mask, 1)
        x = torch.diag(wei).mm(x)
        EncX = x

        # x1 = x1[:, 0]
        # x1 = self.to_latent(x1)

        # EncX = x1
        output = self.Classifier(x)

        # add evidence
        # evidence = F.softplus(self.Classifier(x1))
        # alpha = evidence + 1
        # S = torch.sum(alpha, dim=1, keepdim=True)
        # u = self.class_num / S

        return output, EncX   # alpha, uncertainty, feature_map

def S_model(d_list,
              d_model=768,
              n_layers=2, heads=4, classes_num=10, tau=0.5, dropout=0.2,
              load_weights=None,
              device=torch.device('cuda:0')):
    """
    params: d_list-->list-->dims of view
    d_model--int--num of neurons
    """
    assert d_model % heads == 0
    assert dropout < 1

    model = Model(d_list, d_model, n_layers, heads, classes_num, tau, dropout)

    if load_weights is not None:
        print("loading pretrained weights...")
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    model = model.to(device)

    return model