import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

# 一个基础的self-attention模块
class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    I believe I could have just used torch.nn.MultiheadAttention but their documentation
    is all but absent and code ugly so I don't trust it, rolling my own here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask（因果编码） to ensure that attention is only applied to the left in the input sequence
        # 保证注意力只应用于输入序列的左侧
        # torch.tril 返回一个下三角矩阵，即矩阵中仅包含主对角线及其以下的元素，其余元素为 0
        # view调整形状
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    # 这段代码实现了一个多头自注意力机制的前向传播步骤
    def forward(self, x, layer_past=None):
        # x：输入张量，形状为 (B, T, C)，其中：
        # B 是批量大小（batch size）。
        # T 是序列长度（time steps）。
        # C 是输入特征维度（embedding dimension）
        # 这行代码的作用是从输入张量 x 中解包其尺寸，以便后续在模型中使用
        B, T, C = x.size()

        # transpose是移动数据的维度
        # transpose(1, 2) 将维度从(B, T, nh, hs)转换为 (B, nh, T, hs)，将注意力头维度 nh 移到批次维度之后
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # 用前面定义的nn.linear计算出k q v
        # 其中self.n_head和C // self.n_head体现运用了多头注意力
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # k.shape (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # q.shape (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # v.shape (B, nh, T, hs)

        # 具体运算中，各个头之间相互不影响，因而是序列长度T之中的embeddings在相互计算注意力分数
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # 运用了广播机制！
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # 应用因果掩码，防止模型在预测时看到未来的信息。
        # 将掩码中为 0 的位置填充为一个非常小的值（-1e10），使这些位置的注意力得分接近于 0
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, -1e10) # todo: just use float('-inf') instead?
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        # A 的形状是 (B, nh, T, T)，最后一维是 T。
        # B 的形状是 (B, nh, T, hs)，倒数第二维是 T。
        # 因此，两个张量的形状是兼容的，可以进行矩阵乘法
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # 重新组合成一开始的(B, T, C)形状
        # contiguous() 是 PyTorch 中的一个张量方法，用于返回一个连续的内存布局的张量。
        # 具体来说，张量在内存中存储数据的方式可能是非连续的，
        # 尤其是在进行了某些操作后（如转置、切片、视图变换等）。
        # contiguous() 可以确保返回的张量是以连续的方式存储在内存中的。
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

"""
Write your SynthesizerAttention below.
Hint: paste over the CausalSelfAttention above and modify it minimally.
"""

# 使用一个新的attention
class SynthesizerAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 确保嵌入维度 n_embd 能被头数 n_head 整除
        assert config.n_embd % config.n_head == 0

        # config.n_embd是公式中的d
        # config.block_size-1是公式中的l
        # config.n_head是公式中的n
        # NEW learnable weights
        # w1是公式中的Ai,但是分头的功能在forward中实现
        self.w1 = nn.Linear(config.n_embd, config.n_embd)

        self.b1 = nn.Parameter(torch.zeros(config.n_embd // config.n_head))

        # nn.Parameter(torch.randn(3, 3)) 定义了一个形状为 (3, 3) 的可训练参数
        # 所以这里就是定义了里面这个zero tensor形状的可训练参数
        # w2是公式中的Bi
        self.w2 = nn.Parameter(torch.zeros(config.n_embd // config.n_head,
            config.block_size-1))
        self.b2 = nn.Parameter(torch.zeros(config.block_size-1))
        # value projection
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in
        #     the input sequence
        # causal mask!
        self.register_buffer("mask", torch.tril(
            torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.block_size = config.block_size

        nn.init.uniform_(self.w2,-0.001,0.001)

    def forward(self, x, layer_past=None):
        # TODO [part g]: Write your SynthesizerAttention below.
        #   Do not modify __init__().
        # Hints:
        #   - Paste over the CausalSelfAttention above and modify it minimally.
        #   - Consider especially the parameters self.w1, self.w2 and self.b2.
        #       How do these map to the matrices in the handout?

        B, T, C = x.size()

        # 实现了XAi, 
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        w1 = self.w1(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # w1.shape (B, nh, T, hs) -> (1, 8, 32, 32)
        # print("a", a.shape)

        b1 = self.b1

        # 进行切片操作
        # self.w2 的形状是 (config.n_embd // config.n_head, config.block_size-1)
        # w2[:, :T] 表示取 self.w2 的所有行和前 T 列，其中 T 是输入序列的长度。这样操作后，w2 的形状变为 (hs, T)
        w2 = self.w2[:, :T] # w2.shape (hs, T) -> (32, 127)
        # print("b", b.shape)

        # 同样的，取切片
        b2 = self.b2[:T] # (T) -> (127)
        # print("b2", b2.shape)

        # 这里计算了XVi
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # print(v.shape)

        # Synthesizer Attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (F.relu(w1 + b1) @ w2 + b2)
        # print(att.shape)

        # Mask out the future
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, -1e10) # todo: just use float('-inf') instead?
        att = F.softmax(att, dim=-1)
        # print(att.shape)

        att = self.attn_drop(att)
        # print(att.shape)
        # print(v.shape)

        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y
