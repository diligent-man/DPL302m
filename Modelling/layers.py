import math
import torch
import numpy as np

from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

def no_peak_masks(size, option):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('float32')
    np_mask = Variable(torch.from_numpy(np_mask) == 0)

    if option.cuda:
        np_mask = np_mask.to(option.cuda_device)
    return np_mask

def create_masks(source, target, option):
    source_mask = (source != option.source_pad).unsqueeze(-2)

    if target is not None:
        target_mask = (target != option.target_pad).unsqueeze(-2)
        size = target.size(1)
        np_mask = no_peak_masks(size, option)

        if option.cuda == True:
            target_mask = target_mask.to(option.cuda_device)
        target_mask = target_mask & np_mask  # apply bitwise AND

    else:
        target_mask = None
    return source_mask, target_mask


########################################################################################################################
class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_head = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    @staticmethod
    def __attention(q, k, v, d_k, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(dim0=-2, dim1=-1)) / math.sqrt(d_k)

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            scores = dropout(scores)
        output = torch.matmul(scores, v)
        return output

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # 'view' function is meant to reshape the tensor
        k = self.k_linear(k).view(batch_size, -1, self.h, self.d_head)
        q = self.q_linear(q).view(batch_size, -1, self.h, self.d_head)
        v = self.v_linear(v).view(batch_size, -1, self.h, self.d_head)

        k = k.transpose(dim0=1, dim1=2)
        q = q.transpose(dim0=1, dim1=2)
        v = v.transpose(dim0=1, dim1=2)

        scores = self.__attention(q, k, v, self.d_head, mask, self.dropout)
        concat = scores.transpose(dim0=1, dim1=2).contiguous().view(batch_size, -1, self.d_model)

        output = self.out(concat)
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


########################################################################################################################
class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)


class PositionalEncoder(nn.Module):
  def __init__(self, d_model, max_seq_len=500):
    super().__init__()
    self.d_model = d_model

    # create 'PE' matrix that depends on 'pos' and 'i'
    pe = torch.zeros(max_seq_len, d_model)
    for pos in range(max_seq_len):
        for i in range(0, d_model, 2):
            pe[pos, i] = math.sin(pos / pow(10000, (2 * i) / d_model))
            pe[pos, i+1] = math.cos(pos / pow(10000, (2 * i) / d_model))
    """
    Returns a new tensor with a dimension 
    of size 1 inserted at the specified dimension 'dim'.
    >>> x = torch.tensor([1, 2, 3, 4])
    >>> torch.unsqueeze(x, 0)
    tensor([[ 1,  2,  3,  4]])
    >>> torch.unsqueeze(x, 1)
    tensor([[ 1],
            [ 2],
            [ 3],
            [ 4]])
    """
    pe = pe.unsqueeze(0)

    """
    Adding a buffer to the module.
    This is typically used to register a buffer that should not to be 
    considered a model parameter.
    """
    self.register_buffer('pe', pe)

  def forward(self, x):
    x = x * math.sqrt(self.d_model)
    seq_len = x.size(1)
    x = x + Variable(self.pe[:, :seq_len], requires_grad=False)
    return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model)
        self.attn_2 = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)

    def forward(self, x, encoder_outputs, source_mask, target_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, target_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, encoder_outputs, encoder_outputs, source_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x