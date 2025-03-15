import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from kernel import act_quant, weight_dequant, fp8_gemm

"""
DeepSeek-V3 是一个混合专家模型(Mixture of Experts, MoE)，结合了稀疏门控机制和多头注意力机制。该模型具有以下特点：
1. 分布式计算支持 ：支持模型并行和张量并行
2. 量化支持 ：支持 BF16 和 FP8 数据类型
3. RoPE 位置编码 ：使用旋转位置编码，并支持序列长度扩展
4. 混合专家系统 ：结合路由专家和共享专家
"""

"""
这些是 DeepSeek-V3 推理代码中定义的全局变量，它们对模型的分布式计算和性能优化起着关键作用：
"""
"""
- 功能 ：定义了分布式训练/推理中参与的设备数量
- 默认值 : 1(表示单设备运行)
- 作用 ：在模型并行化时用于确定如何分割模型参数和计算
"""
world_size = 1
"""
- 功能 ：表示当前设备在分布式系统中的序号
- 默认值 : 0(表示第一个或唯一的设备)
- 作用 ：决定当前设备负责处理模型的哪一部分
"""
rank = 0
"""
- 功能 ：定义量化计算中的块大小
- 默认值 : 128
- 作用 ：在权重量化和反量化过程中用于确定处理单元的大小，影响计算效率和内存使用
"""
block_size = 128
"""
- 功能 ：指定通用矩阵乘法(GEMM)的实现方式
- 可选值 :
- "bf16" ：使用 BFloat16 精度
- "fp8" ：使用 8位浮点精度
- 默认值 : "bf16"
- 作用 ：控制线性层计算的精度和性能权衡
"""
gemm_impl: Literal["bf16", "fp8"] = "bf16"
"""
- 功能 ：指定注意力机制的实现方式
- 可选值 :
- "naive" ：标准注意力实现，分别缓存键值
- "absorb" ：优化的注意力实现，使用吸收式计算减少内存使用
- 默认值 : "absorb"
- 作用 ：影响注意力计算的效率和内存使用
"""
attn_impl: Literal["naive", "absorb"] = "absorb"


"""
ModelArgs 类用于定义模型的参数和超参数。它包含许多与模型架构、训练 hyperparameters 和性能优化相关的属性。
这些参数共同定义了DeepSeek-V3模型的架构,包括其混合专家系统(MoE)、多头注意力机制(MLA)以及用于处理长序列的YaRN位置编码扩展技术。
模型使用了稀疏门控MoE架构,可以在保持计算效率的同时提高模型容量。
"""
@dataclass
class ModelArgs:
    """
    Data class for defining model arguments and hyperparameters.

    Attributes:
        max_batch_size (int): Maximum batch size. 
        max_seq_len (int): Maximum sequence length.
        dtype (Literal["bf16", "fp8"]): Data type for computations.
        vocab_size (int): Vocabulary size.
        dim (int): Model dimension.
        inter_dim (int): Intermediate dimension for MLP layers.
        moe_inter_dim (int): Intermediate dimension for MoE layers.
        n_layers (int): Number of transformer layers.
        n_dense_layers (int): Number of dense layers in the model.
        n_heads (int): Number of attention heads.
        n_routed_experts (int): Number of routed experts for MoE layers.
        n_shared_experts (int): Number of shared experts for MoE layers.
        n_activated_experts (int): Number of activated experts in MoE layers.
        n_expert_groups (int): Number of expert groups.
        n_limited_groups (int): Number of limited groups for MoE routing.
        score_func (Literal["softmax", "sigmoid"]): Scoring function for MoE routing.
        route_scale (float): Scaling factor for routing scores.
        q_lora_rank (int): LoRA rank for query projections.
        kv_lora_rank (int): LoRA rank for key-value projections.
        qk_nope_head_dim (int): Dimension for query-key projections without positional embeddings.
        qk_rope_head_dim (int): Dimension for query-key projections with rotary embeddings.
        v_head_dim (int): Dimension for value projections.
        original_seq_len (int): Original sequence length.
        rope_theta (float): Base for rotary positional encoding.
        rope_factor (float): Scaling factor for extended sequence lengths.
        beta_fast (int): Fast beta correction factor.
        beta_slow (int): Slow beta correction factor.
        mscale (float): Scaling factor for extended attention.
    """
    # 基础配置参数
    max_batch_size: int = 8 #模型处理的最大批次大小
    max_seq_len: int = 4096 * 4 #模型支持的最大序列长度，设置为16384 (4096×4)
    dtype: Literal["bf16", "fp8"] = "bf16" #计算使用的数据类型，默认为 bfloat16
    vocab_size: int = 102400 #词汇表大小，约10万个token
    dim: int = 2048 #模型的隐藏维度
    inter_dim: int = 10944 #MLP层的中间维度
    moe_inter_dim: int = 1408 #MoE层的中间维度
    n_layers: int = 27 #Transformer层数
    n_dense_layers: int = 1 #密集层数量
    n_heads: int = 16 #注意力头数量
    # MoE (混合专家) 相关参数
    n_routed_experts: int = 64 #路由专家的数量
    n_shared_experts: int = 2 #共享专家的数量
    n_activated_experts: int = 6 #每个输入激活的专家数量
    n_expert_groups: int = 1 #专家组的数量
    n_limited_groups: int = 1 #MoE路由的限制组数
    score_func: Literal["softmax", "sigmoid"] = "softmax" #MoE路由的评分函数
    route_scale: float = 1. #路由分数的缩放因子
    # MLA (多头注意力) 相关参数
    q_lora_rank: int = 0 #查询投影的LoRA秩
    kv_lora_rank: int = 512 #键值投影的LoRA秩
    qk_nope_head_dim: int = 128 #无位置嵌入的查询-键投影维度
    qk_rope_head_dim: int = 64 #带旋转位置嵌入的查询-键投影维度
    v_head_dim: int = 128 #值投影的维度
    # YaRN (位置编码扩展) 相关参数
    original_seq_len: int = 4096 #原始序列长度
    rope_theta: float = 10000.0 #旋转位置编码的基数
    rope_factor: float = 40 #扩展序列长度的缩放因子
    beta_fast: int = 32 #快速β校正因子
    beta_slow: int = 1 #慢速β校正因子
    mscale: float = 1. #扩展注意力的缩放因子

"""
ParallelEmbedding 类是一个支持分布式计算的嵌入层实现，它继承自 PyTorch 的 nn.Module 。
这个类的主要目的是在多设备环境下高效地处理大型词汇表的嵌入操作。
这个类实现了词汇表的分片处理，将整个词汇表分割到多个计算设备上，每个设备只负责处理一部分词汇表的嵌入操作，从而实现并行计算
"""
class ParallelEmbedding(nn.Module):
    """
    Embedding layer with parallelism support across distributed processes.

    - 确保词汇表大小能被设备数量整除
    - 计算每个设备负责的词汇表部分大小 part_vocab_size
    - 确定当前设备负责的词汇表范围 ( vocab_start_idx 到 vocab_end_idx )
    - 创建嵌入权重参数，但只包含当前设备负责的部分

    Args:
        vocab_size (int): Vocabulary size. 词汇表的总大小
        dim (int): Embedding dimension. 嵌入向量的维度
    """
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        assert vocab_size % world_size == 0, f"Vocabulary size must be divisible by world size (world_size={world_size})"
        self.part_vocab_size = (vocab_size // world_size)
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

    """
    前向传播方法的工作流程：

    1. 分布式处理判断 ：首先检查是否在多设备环境下运行 ( world_size > 1 )
    2. 创建掩码 ：创建一个布尔掩码，标记出不属于当前设备负责范围的词汇索引
    3. 索引调整 ：将输入索引减去起始索引，使其适配当前设备的嵌入表
    4. 无效索引处理 ：将掩码标记的索引设为0，避免越界访问
    5. 嵌入查找 ：使用调整后的索引在当前设备的嵌入表中查找对应的嵌入向量
    6. 结果合并 ：在多设备环境下，将掩码位置的嵌入设为0，然后使用 all_reduce 操作合并所有设备的结果
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for parallel embedding layer.

        Args:
            x (torch.Tensor): Input tensor containing token indices.

        Returns:
            torch.Tensor: Embedded representations.

        Raises:
            ValueError: If `world_size` is not defined.
        """
        if world_size > 1:
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x = x - self.vocab_start_idx
            x[mask] = 0
        y = F.embedding(x, self.weight)
        if world_size > 1:
            y[mask] = 0
            dist.all_reduce(y)
        return y

"""
这段代码定义了一个名为 linear 的函数，它是 DeepSeek-V3 模型中的一个关键组件，用于实现线性变换操作，同时支持不同的量化策略
linear 函数实现了标准的线性变换：y = xA^T + b，但增加了对量化权重的支持，使模型能够在不同精度下高效运行。

1. 混合精度支持 ：函数能够处理不同精度的权重，包括全精度、BF16 和 FP8
2. 量化感知计算 ：针对量化权重，使用专门的反量化和矩阵乘法函数
3. 性能优化 ：通过量化技术减少内存使用和提高计算效率
4. 灵活性 ：通过全局变量 gemm_impl 控制计算策略，便于在不同场景下切换
这个函数是 DeepSeek-V3 模型中实现高效推理的关键组件之一，通过量化技术显著减少了模型的内存占用和计算开销，同时保持了模型的精度。
"""
def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Applies a linear transformation to the incoming data: y = xA^T + b.
    This function supports specialized implementations based on quantization
    and tensor formats.

    Args:
        x (torch.Tensor): The input tensor. 输入张量，包含需要进行线性变换的数据
        weight (torch.Tensor): The weight tensor. It may be quantized and 
            requires dequantization for certain cases. 权重张量，可能是量化过的（低精度存储）
        bias (Optional[torch.Tensor]): The bias tensor to be added. Default is None. 偏置张量（可选），默认为 None

    Returns:
        torch.Tensor: The result of the linear transformation, which may involve 
        quantization-aware computations depending on the input parameters.

    Notes:
        - If `weight` is quantized (e.g., `element_size() == 1`), a dequantized version 
          is used for computation.
        - If `gemm_impl == "bf16"`, dequantization and a `bf16` GEMM operation are applied.
        - For other cases, the function applies quantization to `x` and uses `fp8_gemm` for computation.
    """
    #当权重不是量化格式时（每个元素占用空间 > 1 字节），直接使用 PyTorch 的标准线性变换函数。
    if weight.element_size() > 1:
        return F.linear(x, weight, bias)
    #当权重是量化格式且全局设置为 BF16 时，先对权重进行反量化（weight_dequant），然后使用标准线性变换。
    elif gemm_impl == "bf16":
        weight = weight_dequant(weight, weight.scale)
        return F.linear(x, weight, bias)
    #当使用 FP8 精度时，先对输入 x 进行量化（act_quant），然后使用专门的 fp8_gemm 函数进行矩阵乘法，最后添加偏置（如果有）。
    else:
        x, scale = act_quant(x, block_size)
        y = fp8_gemm(x, scale, weight, weight.scale)
        if bias is not None:
            y += bias
        return y

"""
这段代码定义了一个名为 Linear 的自定义线性层类，它是 DeepSeek-V3 模型中的一个核心组件，继承自 PyTorch 的 nn.Module 。
这个类实现了支持量化权重的线性变换操作。

dtype = torch.bfloat16 ：类级别的默认数据类型，设置为 BFloat16 格式，这是一种在深度学习中常用的低精度浮点格式，可以减少内存使用并加速计算。
"""
class Linear(nn.Module):
    """
    Custom linear layer with support for quantized weights and optional bias.

    Args:
        in_features (int): Number of input features. 输入特征的维度
        out_features (int): Number of output features. 输出特征的维度
        bias (bool): Whether to include a bias term. Defaults to False. 是否使用偏置项，默认为 False
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`. 数据类型，默认使用类属性 Linear.dtype
    """
    dtype = torch.bfloat16

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        #创建权重参数：
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or Linear.dtype))
        #对于量化权重（当元素大小为1字节时）创建缩放因子：
        if self.weight.element_size() == 1:
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float32))
        #对于非量化权重，注册一个空的 scale 参数
        else:
            self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the custom linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor after linear computation.
        """
        return linear(x, self.weight, self.bias)

"""
ColumnParallelLinear 类是 DeepSeek-V3 模型中实现分布式计算的关键组件之一，它继承自前面定义的 Linear 类，专门用于实现列并行的线性变换操作。

这个类的主要目的是将线性层的输出特征（列）分割到多个计算设备上，从而实现模型的并行计算。
这种并行策略对于大型模型尤其重要，因为它可以有效地分散计算负载和内存使用。

ColumnParallelLinear 主要用于：
1. 注意力机制中的查询、键、值投影
2. MLP 层中的输入到隐藏层的变换
3. MoE（混合专家）模型中的专家路由
"""
class ColumnParallelLinear(Linear):
    """
    Linear layer with column parallelism, splitting output features across distributed processes.

    初始化过程包含以下关键步骤：

    1. 参数验证 ：首先确保输出特征的数量能被设备数量（ world_size ）整除，这是列并行的基本要求
    2. 计算每个设备的输出特征数 ： self.part_out_features = out_features // world_size
    3. 调用父类初始化 ：使用原始输入特征数和计算得到的部分输出特征数初始化父类（ Linear ）
    这种设计确保每个设备只负责处理总输出特征的一部分，从而实现计算的分布式处理。

    Args:
        in_features (int): Number of input features.
        out_features (int): Total number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert out_features % world_size == 0, f"Output features must be divisible by world size (world_size={world_size})"
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for column parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with column-parallel computation.
        """
        y = linear(x, self.weight, self.bias)
        return y

"""
RowParallelLinear 类是 DeepSeek-V3 模型中实现行并行计算的关键组件，它继承自基础的 Linear 类，专门用于实现输入特征的行并行分割。

这个类的主要目的是将线性层的输入特征（行）分割到多个计算设备上，从而实现模型的并行计算。
这种并行策略与 ColumnParallelLinear 相辅相成，共同构成了模型的分布式计算框架。

RowParallelLinear 主要用于：
1. 注意力层的输出投影
2. MLP 层的第二个线性变换
3. 需要将分布式计算结果合并的场景
"""
class RowParallelLinear(Linear):
    """
    Linear layer with row parallelism, splitting input features across distributed processes.

    初始化过程包含以下关键步骤：

    1. 参数验证 ：确保输入特征的数量能被设备数量（ world_size ）整除
    2. 计算每个设备的输入特征数 ： self.part_in_features = in_features // world_size
    3. 调用父类初始化 ：使用计算得到的部分输入特征数和完整的输出特征数初始化父类

    Args:
        in_features (int): Total number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert in_features % world_size == 0, f"Input features must be divisible by world size (world_size={world_size})"
        self.part_in_features = in_features // world_size
        super().__init__(self.part_in_features, out_features, bias, dtype)

    """
    前向传播的工作流程：
    1. 线性变换 ：使用 linear 函数进行基础的线性变换
    2. 结果聚合 ：当在多设备环境下时，使用 all_reduce 操作合并所有设备的计算结果
    3. 偏置添加 ：如果存在偏置项，将其加到结果上
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for row parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with row-parallel computation.
        """
        y = linear(x, self.weight)
        if world_size > 1:
            dist.all_reduce(y)
        if self.bias is not None:
            y += self.bias
        return y

"""
RMSNorm 类是 DeepSeek-V3 模型中的一个关键组件，用于对输入特征进行归一化处理，以增强模型的泛化能力。

这段代码定义了一个名为 RMSNorm 的层归一化类，它是 Root Mean Square Layer Normalization 的实现。
RMSNorm 是 LayerNorm 的一个变体，具有更简单的计算过程和更好的性能。
"""
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Args:
        dim (int): Dimension of the input tensor.
        eps (float): Epsilon value for numerical stability. Defaults to 1e-6.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)


def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args (ModelArgs): Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """
        Computes the correction dimension for a given number of rotations in the rotary positional embedding.

        Args:
            num_rotations (float): Number of rotations to compute the correction for.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            float: The correction dimension based on the input parameters.
        """
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """
        Computes the range of correction dimensions for rotary positional embeddings.

        Args:
            low_rot (float): Lower bound for the number of rotations.
            high_rot (float): Upper bound for the number of rotations.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            Tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min, max, dim):
        """
        Computes a linear ramp function used to smooth values between a minimum and maximum range.

        Args:
            min (float): Minimum value for the ramp function.
            max (float): Maximum value for the ramp function.
            dim (int): Dimensionality of the ramp tensor.

        Returns:
            torch.Tensor: A tensor of shape (dim,) with values linearly interpolated between 0 and 1,
                clamped to the range [0, 1].
        """
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)


class MLA(nn.Module):
    """
    Multi-Headed Attention Layer (MLA).

    Attributes:
        dim (int): Dimensionality of the input features. 输入特征维度
        n_heads (int): Number of attention heads. 注意力头数量
        n_local_heads (int): Number of local attention heads for distributed systems. 每个设备上的本地注意力头数量
        q_lora_rank (int): Rank for low-rank query projection. 查询投影的 LoRA 秩
        kv_lora_rank (int): Rank for low-rank key/value projection. 键值投影的 LoRA 秩
        qk_nope_head_dim (int): Dimensionality of non-positional query/key projections.
        qk_rope_head_dim (int): Dimensionality of rotary-positional query/key projections.
        qk_head_dim (int): Total dimensionality of query/key projections.
        v_head_dim (int): Dimensionality of value projections.
        softmax_scale (float): Scaling factor for softmax in attention computation.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        if self.q_lora_rank == 0:
            # 非LoRA的权重矩阵大小：(dim, n_heads * qk_head_dim) 
            # 以671B为例，矩阵大小为 (7168, 128 * 192) = 176,160,768
            self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)
        else:
            # 第一个LoRA的权重矩阵大小：(dim, q_lora_rank)
            # 以671B为例，第一个LoRA矩阵大小为 (7168, 1536) = 11,010,048
            self.wq_a = Linear(self.dim, self.q_lora_rank)
            self.q_norm = RMSNorm(self.q_lora_rank)
            # 第二个LoRA的权重矩阵大小：(q_lora_rank, n_heads * qk_head_dim)
            # 以671B为例，第二个LoRA矩阵大小为 (1536, 128 * 192) = 37,748,736
            self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
        self.softmax_scale = self.qk_head_dim ** -0.5
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        if attn_impl == "naive":
            self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.qk_head_dim), persistent=False)
            self.register_buffer("v_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.v_head_dim), persistent=False)
        else:
            self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank), persistent=False)
            self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        """
        Forward pass for the Multi-Headed Attention Layer (MLA).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            start_pos (int): Starting position in the sequence for caching.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        if attn_impl == "naive":
            q = torch.cat([q_nope, q_pe], dim=-1)
            kv = self.wkv_b(self.kv_norm(kv))
            kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
            self.k_cache[:bsz, start_pos:end_pos] = k
            self.v_cache[:bsz, start_pos:end_pos] = v
            scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos]) * self.softmax_scale
        else:
            wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size) 
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
            self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
            self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
            scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
                      torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale
        if mask is not None:
            scores += mask.unsqueeze(1)
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        if attn_impl == "naive":
            x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])
        else:
            x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
        x = self.wo(x.flatten(2))
        return x

"""
标准多层感知机，用于密集层的前馈网络
继承自 PyTorch 的 nn.Module 基类,这是 PyTorch 中创建神经网络模块的标准方式

1. SwiGLU 激活函数 ：使用了比传统 GELU 或 ReLU 更高效的 SwiGLU 变体，通过门控机制提高模型表达能力
2. 并行计算优化 :
   - ColumnParallelLinear 将输出特征分割到不同设备
   - RowParallelLinear 将输入特征分割到不同设备
   - 这种设计使大规模模型能够在多 GPU 环境下高效训练和推理
3. 计算效率 ：通过矩阵乘法和元素级操作的组合，在保持表达能力的同时提高计算效率
这种 MLP 设计是现代大型语言模型中的标准组件，特别是在 Transformer 架构中作为每个块的前馈网络部分。
"""
class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) used as a feed-forward layer.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    """
    初始化方法接收两个参数：
    - dim : 输入和输出的维度
    - inter_dim : 中间隐藏层的维度
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the MLP layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        """
        - 创建第一个线性变换层 w1
        - 使用 ColumnParallelLinear 而不是标准的 nn.Linear ，支持模型并行
        - 将输入从 dim 维度映射到 inter_dim 维度
        - 列并行意味着输出特征被分割到不同的设备上

        输入x的形状为 (batch_size, dim)
        每个设备的weight的形状为 (dim, inter_dim // world_size)
        每个设备的输出y的形状为 (batch_size, inter_dim // world_size)
        """
        self.w1 = ColumnParallelLinear(dim, inter_dim)
        """
        - 创建第二个线性变换层 w2
        - 使用 RowParallelLinear ，同样支持模型并行
        - 将中间表示从 inter_dim 维度映射回 dim 维度
        - 行并行意味着输入特征被分割到不同的设备上

        输入x的形状为 (batch_size, inter_dim // world_size)
        每个设备的weight的形状为 (inter_dim // world_size, dim)
        每个设备的输出y的形状为 (batch_size, dim)，并使用all_reduce，将所有设备上的数据求和，最终y的形状仍然为 (batch_size, dim)
        """
        self.w2 = RowParallelLinear(inter_dim, dim)
        """
        - 创建第三个线性变换层 w3
        - 与 w1 类似，也是从 dim 映射到 inter_dim
        - 这是 SwiGLU 激活函数所需的额外线性变换

        输入x的形状为 (batch_size, dim)
        每个设备的weight的形状为 (dim, inter_dim // world_size)
        每个设备的输出y的形状为 (batch_size, inter_dim // world_size)
        """
        self.w3 = ColumnParallelLinear(dim, inter_dim)

    """
    - 定义模型的前向传播逻辑
    - 实现了 SwiGLU 激活函数的变体，计算过程如下：
    1. self.w1(x) : 将输入 x 通过第一个线性层变换
    2. F.silu(self.w1(x)) : 对结果应用 SiLU (Sigmoid Linear Unit) 激活函数
    3. self.w3(x) : 将输入 x 通过第三个线性层变换
    4. F.silu(self.w1(x)) * self.w3(x) : 将两个结果进行元素级乘法
    5. self.w2(...) : 将乘法结果通过第二个线性层变换，得到最终输出
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after MLP computation.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

"""
Gate 类是 DeepSeek-V3 模型中混合专家系统(Mixture of Experts, MoE)的核心组件，负责实现专家路由机制。
它决定输入数据应该被发送到哪些专家模型进行处理，是实现条件计算和动态路由的关键部分。

Gate 类的主要功能是根据输入特征计算路由分数，并选择最合适的专家来处理每个输入样本。
"""
class Gate(nn.Module):
    """
    Gating mechanism for routing inputs in a mixture-of-experts (MoE) model.

    Attributes:
        dim (int): Dimensionality of input features. 输入特征的维度
        topk (int): Number of top experts activated for each input. 每个输入激活的专家数量（由 args.n_activated_experts 指定）
        n_groups (int): Number of groups for routing. 路由分组数量，用于分层路由
        topk_groups (int): Number of groups to route inputs to. 要路由输入的组数
        score_func (str): Scoring function ('softmax' or 'sigmoid'). 评分函数类型，可以是 "softmax" 或 "sigmoid"
        route_scale (float): Scaling factor for routing weights. 路由权重的缩放因子
        weight (torch.nn.Parameter): Learnable weights for the gate. 可学习的门控权重参数，形状为 [n_routed_experts, dim]
        bias (Optional[torch.nn.Parameter]): Optional bias term for the gate. 可选的偏置项，仅当 dim == 7168 时使用
    """
    def __init__(self, args: ModelArgs):
        """
        Initializes the Gate module.

        Args:
            args (ModelArgs): Model arguments containing gating parameters.
        """
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts)) if self.dim == 7168 else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Routing weights and selected expert indices.
        """
        #使用线性变换计算每个输入对每个专家的初始分数
        #以671B未例子，x的矩阵大小为 (batch_size, 7168)，self.weight的矩阵大小为 (256, 7168)
        #x * W.transpose(0, 1) = (batch_size, 256)  scores既256个专家的分数
        scores = linear(x, self.weight)
        #根据配置使用 softmax 或 sigmoid 函数将分数转换为概率
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias

        """
        分组路由机制主要解决以下几个问题：

        1. 计算效率 ：对于拥有大量专家的模型，直接从所有专家中选择会导致计算开销过大。分组可以将搜索空间分层，先选择有潜力的组，再在组内选择专家。
        2. 负载均衡 ：分组可以更好地平衡专家的使用率，避免少数专家被过度使用（"专家崩溃"问题），提高模型整体的利用率。
        3. 专业化 ：不同组的专家可以专注于不同类型的输入，形成更有效的专业化分工。
        4. 扩展性 ：当模型规模扩大时，分组机制使得添加更多专家变得更加高效和可管理。
        """
        if self.n_groups > 1:
            # 将专家分数重塑为 [batch_size, n_groups, experts_per_group] 的形状
            scores = scores.view(x.size(0), self.n_groups, -1)
            # 计算每个组的总体得分
            if self.bias is None:
                # 如果没有偏置，使用每组中最大的专家分数作为组分数
                group_scores = scores.amax(dim=-1)
            else:
                # 如果有偏置，使用每组中得分最高的两个专家分数之和作为组分数
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            # 选择得分最高的 topk_groups 个组的索引
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            # 创建掩码，标记未被选中的组
            # scatter_: 在索引 indices 处将值设为 False，其他位置保持为 True
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
            # 将未选中组的专家分数设为负无穷，确保它们不会被选中
            # 然后将三维张量重新展平为二维 [batch_size, n_groups * experts_per_group]
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
        #为每个输入选择得分最高的 topk 个专家的索引
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        #根据选定的专家索引，从原始分数中收集对应的权重
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights.type_as(x), indices

"""
Expert 类是 DeepSeek-V3 模型中混合专家系统(Mixture of Experts, MoE)的核心组件之一，它实现了单个"专家"的功能。
在 MoE 架构中，多个专家并行工作，每个专家负责处理特定类型的输入。

Expert 类继承自 PyTorch 的 nn.Module ，
实现了一个前馈神经网络，其结构类似于标准的 MLP (多层感知机)，但专门用于 MoE 架构中。

## 与 MLP 类的区别
虽然 Expert 和 MLP 类的结构相似，但有几个关键区别:

1. Expert 使用普通的 Linear 层，而 MLP 使用分布式的 ColumnParallelLinear 和 RowParallelLinear
2. Expert 设计为 MoE 架构的一部分，由 Gate 类动态路由输入
3. 每个输入只会激活少数几个 Expert ，而 MLP 处理所有输入

## 在 MoE 中的作用
在 MoE 架构中， Expert 类的实例被组织在 MoE 类中，通过 Gate 类实现的路由机制，
每个输入只会被发送到少数几个专家进行处理。这种稀疏激活的设计使模型能够:

1. 增加模型容量而不显著增加计算成本
2. 使不同专家专注于不同类型的输入
3. 提高模型处理复杂任务的能力
"""
class Expert(nn.Module):
    """
    Expert layer for Mixture-of-Experts (MoE) models.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the Expert layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = Linear(dim, inter_dim)
        self.w2 = Linear(inter_dim, dim)
        self.w3 = Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Expert layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert computation.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

"""
混合专家系统（Mixture of Experts, MoE）模块。这是一种稀疏激活的神经网络架构，
通过动态路由机制将输入分配给不同的"专家"子网络处理，从而在不显著增加计算成本的情况下提高模型容量。

## MoE 的关键特性
1. 稀疏激活 ：每个输入只激活少数几个专家（由 n_activated_experts 控制），大大减少了计算量
2. 分布式计算 ：专家被分布在多个计算设备上，每个设备只负责一部分专家的计算
3. 负载均衡 ：通过 Gate 类实现的路由机制，动态决定输入应该由哪些专家处理
4. 共享专家 ：除了路由专家外，还有共享专家处理所有输入，确保模型的基础能力
"""
class MoE(nn.Module):
    """
    Mixture-of-Experts (MoE) module.

    Attributes:
        dim (int): Dimensionality of input features.
        n_routed_experts (int): Total number of experts in the model.
        n_local_experts (int): Number of experts handled locally in distributed systems.
        n_activated_experts (int): Number of experts activated for each input.
        gate (nn.Module): Gating mechanism to route inputs to experts.
        experts (nn.ModuleList): List of expert modules.
        shared_experts (nn.Module): Shared experts applied to all inputs.
    """
    def __init__(self, args: ModelArgs):
        """
        Initializes the MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
        """
        super().__init__()
        self.dim = args.dim
        assert args.n_routed_experts % world_size == 0, f"Number of experts must be divisible by world size (world_size={world_size})"
        self.n_routed_experts = args.n_routed_experts
        # 计算每个设备负责的本地专家数量 n_local_experts
        self.n_local_experts = args.n_routed_experts // world_size
        self.n_activated_experts = args.n_activated_experts
        # 确定当前设备负责的专家索引范围（ experts_start_idx 到 experts_end_idx ）
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        # 创建路由门控机制 gate ，负责决定输入应该被发送到哪些专家
        self.gate = Gate(args)
        # 创建专家列表 experts ，但只实例化当前设备负责的专家（其他位置为 None）
        self.experts = nn.ModuleList([Expert(args.dim, args.moe_inter_dim) if self.experts_start_idx <= i < self.experts_end_idx else None
                                      for i in range(self.n_routed_experts)])
        # 创建共享专家 shared_experts ，这是一个标准的 MLP，会处理所有输入
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
        """
        shape = x.size()
        x = x.view(-1, self.dim)
        # 通过 gate 获取路由权重和专家索引,weights 表示每个输入对应的专家权重,indices 表示每个输入应该路由到哪些专家
        weights, indices = self.gate(x)
        y = torch.zeros_like(x)
        # 统计每个专家被选中的次数
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        # 只处理当前设备负责的专家（从 experts_start_idx 到 experts_end_idx ）
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            # 对于每个专家，找出应该由它处理的输入索引
            idx, top = torch.where(indices == i)
            # 将专家的输出乘以对应的权重，累加到结果张量 y 中
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        z = self.shared_experts(x)
        # 在分布式环境中，使用 all_reduce 操作合并所有设备的专家输出
        if world_size > 1:
            dist.all_reduce(y)
        return (y + z).view(shape)

"""
Block 类是 DeepSeek-V3 模型中的一个核心组件，它实现了 Transformer 架构中的基本构建块。
每个 Block 包含一个注意力层和一个前馈网络层，这是现代大型语言模型的标准结构。

Block 类继承自 PyTorch 的 nn.Module ，包含以下主要组件：

1. 注意力层 (attn) : 实现为 MLA （多头注意力）类的实例
2. 前馈网络 (ffn) : 根据层位置不同，可以是 MLP （多层感知机）或 MoE （混合专家）
3. 层归一化 (attn_norm, ffn_norm) : 两个 RMSNorm 实例，分别用于注意力层和前馈网络的输入归一化
"""
class Block(nn.Module):
    """
    Transformer block combining attention and feed-forward layers.

    Attributes:
        attn (nn.Module): Attention layer (MLA).
        ffn (nn.Module): Feed-forward network (MLP or MoE).
        attn_norm (nn.Module): Layer normalization for attention.
        ffn_norm (nn.Module): Layer normalization for feed-forward network.
    """
    def __init__(self, layer_id: int, args: ModelArgs):
        """
        Initializes the Transformer block.

        Args:
            layer_id (int): Layer index in the transformer. 表示当前块在 Transformer 中的层索引
            args (ModelArgs): Model arguments containing block parameters. 包含模型参数的 ModelArgs 实例
        """
        super().__init__()
        self.attn = MLA(args)
        # 当 layer_id < args.n_dense_layers 时，使用标准的 MLP,否则使用 MoE （混合专家系统）
        # 这种设计允许模型在浅层使用密集计算，而在深层使用更高效的稀疏专家混合，平衡了计算效率和模型表达能力。
        self.ffn = MLP(args.dim, args.inter_dim) if layer_id < args.n_dense_layers else MoE(args)
        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position in the sequence.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor after block computation.
        """
        """
        注意力计算 :
        - 对输入 x 应用层归一化 ( self.attn_norm(x) )
        - 将归一化后的输入传递给注意力层 ( self.attn(...) )
        - 将注意力层的输出与原始输入相加（残差连接）
        """
        x = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)
        """
        前馈网络计算 :

        - 对上一步的结果应用层归一化 ( self.ffn_norm(x) )
        - 将归一化后的输入传递给前馈网络 ( self.ffn(...) )
        - 将前馈网络的输出与上一步的结果相加（第二个残差连接）
        """
        x = x + self.ffn(self.ffn_norm(x))
        return x

"""
Transformer 类继承自 PyTorch 的 nn.Module ，包含了以下主要组件：

1. 词嵌入层 ( embed )
2. Transformer 层序列 ( layers )
3. 最终层归一化 ( norm )
4. 输出投影层 ( head )
5. 位置编码 ( freqs_cis )
"""
class Transformer(nn.Module):
    """
    Transformer model with positional embeddings, multiple layers, and output projection.

初始化过程中首先设置了分布式环境变量：

- 获取当前分布式环境的设备总数 ( world_size ) 和当前设备的序号 ( rank )
- 根据配置参数设置线性层的数据类型，支持 FP8 或 BF16 精度
然后创建模型的各个组件：

- 设置最大序列长度
- 创建并行嵌入层，用于将输入 token 转换为向量表示
- 创建多个 Transformer 层，每层是一个 Block 实例
- 创建最终的层归一化和输出投影层
- 预计算并注册旋转位置编码 ( freqs_cis )

    Attributes:
        max_seq_len (int): Maximum sequence length for the transformer.
        embed (nn.Module): Embedding layer for input tokens.
        layers (torch.nn.ModuleList): List of transformer blocks.
        norm (nn.Module): Layer normalization applied after all blocks.
        head (nn.Module): Output projection layer mapping to vocabulary size.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
    """
    def __init__(self, args: ModelArgs):
        """
        Initializes the Transformer model.

        Args:
            args (ModelArgs): Model arguments containing transformer parameters.
        """
        global world_size, rank
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        Linear.dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        super().__init__()
        self.max_seq_len = args.max_seq_len
        self.embed = ParallelEmbedding(args.vocab_size, args.dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(Block(layer_id, args))
        self.norm = RMSNorm(args.dim)
        self.head = ColumnParallelLinear(args.dim, args.vocab_size, dtype=torch.get_default_dtype())
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        """
        Forward pass for the Transformer model.

前向传播方法使用了 @torch.inference_mode() 装饰器，表明该方法仅用于推理，不会计算梯度。

处理流程如下：

1. 获取输入序列长度 seqlen
2. 通过嵌入层将输入 token 转换为向量表示 h
3. 从预计算的位置编码中获取当前序列对应的部分
4. 对于长度大于 1 的序列，创建注意力掩码 mask ，实现自回归生成
   - 掩码是一个上三角矩阵，对角线以上的值为负无穷，确保每个位置只能看到自己及之前的位置
5. 依次通过每个 Transformer 层，更新隐藏状态 h
6. 对最终的隐藏状态应用层归一化，并只保留最后一个位置的表示 [:, -1]
7. 通过输出投影层将隐藏状态映射到词汇表大小的 logits
8. 在分布式环境中，收集并合并所有设备的 logits

## 分布式计算支持
代码中包含了多处分布式计算的支持：

1. 初始化时设置全局的 world_size 和 rank 变量
2. 使用 ParallelEmbedding 和 ColumnParallelLinear 等分布式层
3. 在前向传播的最后使用 dist.all_gather 收集所有设备的输出结果
## 性能优化
代码中包含了多种性能优化技术：

1. 使用低精度计算 (FP8 或 BF16)，减少内存使用和提高计算速度
2. 预计算位置编码，避免重复计算
3. 使用 persistent=False 标记缓冲区，优化内存使用
4. 使用 inference_mode() 而非 no_grad() ，进一步优化推理性能
这个 Transformer 类是 DeepSeek-V3 模型的核心实现，它结合了现代 Transformer 架构的多项技术创新，包括旋转位置编码、混合专家系统、分布式计算和量化推理等，使模型能够高效地处理长序列并在多设备环境中运行。

        Args:
            tokens (torch.Tensor): Input tensor of token IDs with shape (batch_size, seq_len).
            start_pos (int, optional): Starting position in the sequence for rotary embeddings. Defaults to 0.

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, vocab_size).
        """
        seqlen = tokens.size(1)
        h = self.embed(tokens)
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(1)
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)[:, -1]
        # self.head = ColumnParallelLinear(args.dim, args.vocab_size, dtype=torch.get_default_dtype())
        # 使用 ColumnParallelLinear ，支持模型并行
        # 将输入从 dim 维度映射到 vocab_size 维度
        # 行并行意味着输入特征被分割到不同的设备上
        # 输入h的形状为 (batch_size, dim)
        # 每个设备的weight的形状为 (dim, vocab_size // world_size)
        # 每个设备的输出y的形状为 (batch_size, vocab_size // world_size)
        # 使用all_gather，收集每个设备的输出，并将它们拼接在一起，最终y的形状为 (batch_size, vocab_size)
        logits = self.head(h)
        if world_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(world_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)
        return logits


if __name__ == "__main__":
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    args = ModelArgs()
    x = torch.randint(0, args.vocab_size, (2, 128))
    model = Transformer(args)
    print(model(x).size())
