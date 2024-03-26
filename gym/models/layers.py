import torch
import torch.nn as nn
from torch.nn import functional as F
import math

from typing import Union
from dataclasses import dataclass
from einops import rearrange, repeat, einsum
from mamba_ssm import Mamba


# just for compatibility with other files, we do not use this Convolution module
class Convolution(nn.Module):
    def __init__(self, config, hidden_size, block_index):
        super().__init__()
        self.window_size = config.window_size
        self.remove_act_embs = config.remove_act_embs

        self.rtg_conv1d = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=self.window_size, groups=hidden_size)
        self.obs_conv1d = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=self.window_size, groups=hidden_size)
        if not self.remove_act_embs:
            self.act_conv1d = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=self.window_size, groups=hidden_size)

    def forward(self, x):
        #window_size = self.window_size

        # pad the input tensor with zeros along the sequence dimension
        padded_tensor = torch.nn.functional.pad(x, (0, 0, self.window_size - 1, 0)).transpose(1, 2)

        if not self.remove_act_embs:
            rtg_conv_tensor = self.rtg_conv1d(padded_tensor)[:, :, ::3]
            obs_conv_tensor = self.obs_conv1d(padded_tensor)[:, :, 1::3]
            act_conv_tensor = self.act_conv1d(padded_tensor)[:, :, 2::3]

            conv_tensor = torch.cat((rtg_conv_tensor.unsqueeze(3), obs_conv_tensor.unsqueeze(3), act_conv_tensor.unsqueeze(3)), dim=3)
            conv_tensor = conv_tensor.reshape(conv_tensor.shape[0], conv_tensor.shape[1], -1)

        else:
            rtg_conv_tensor = self.rtg_conv1d(padded_tensor)[:, :, ::2]
            obs_conv_tensor = self.obs_conv1d(padded_tensor)[:, :, 1::2]

            conv_tensor = torch.cat((rtg_conv_tensor.unsqueeze(3), obs_conv_tensor.unsqueeze(3)), dim=3)
            conv_tensor = conv_tensor.reshape(conv_tensor.shape[0], conv_tensor.shape[1], -1)

        conv_tensor = conv_tensor.transpose(1, 2)  #.to('cuda')
        return conv_tensor


#********** **** Mamba mini **** **********
# Below shows the essential functionalities inside `Mamba` block imported from `mamba_ssm`
# Practically no need to use this MambaBlock, just for readability and the very first confirmation

# source: https://github.com/johnma2006/mamba-minimal
"""Simple, minimal implementation of Mamba.

Refs:
    [1] Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Albert Gu and Tri Dao)
        https://arxiv.org/abs/2312.00752
    [2] The Annotated S4 (Sasha Rush and Sidd Karamcheti)
        https://srush.github.io/annotated-s4

Glossary:
    b: batch size                       (`B` in Mamba paper [1] Algorithm 2)
    l: sequence length                  (`L` in [1] Algorithm 2)
    d or d_model: hidden dim
    n or d_state: latent state dim      (`N` in [1] Algorithm 2)
    expand: expansion factor            (`E` in [1] Section 3.4)
    d_in or d_inner: d * expand         (`D` in [1] Algorithm 2)
    A, B, C, D: state space parameters  (See any state space representation formula)
                                        (B, C are input-dependent (aka selective, a key innovation in Mamba); A, D are not)
    Δ or delta: input-dependent step size
    dt_rank: rank of Δ                  (See [1] Section 3.6 "Parameterization of ∆")
"""

@dataclass
class ModelArgs:
    d_model: int
    #n_layer: int
    #vocab_size: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

        #if self.vocab_size % self.pad_vocab_size_multiple != 0:
        #    self.vocab_size += (self.pad_vocab_size_multiple
        #                        - self.vocab_size % self.pad_vocab_size_multiple)


class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1].

        Note: the official repo chains residual blocks that look like
            [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> ...
        where the first Add is a no-op. This is purely for performance reasons as this
        allows them to fuse the Add->Norm.

        We instead will realize our blocks as the more familiar, simpler, and numerically equivalent
            [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> ....
        """
        super().__init__()
        self.args = args

        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)

        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)

        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)


        # ===== added from the original: see https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L82
        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = args.dt_rank**-0.5 * args.dt_scale
        if args.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif args.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(torch.rand(args.d_inner) * (math.log(args.dt_max) - math.log(args.dt_min)) + math.log(args.dt_min)).clamp(min=args.dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True
        # ===== =====


        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)


    def forward(self, x):
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].

        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d)

        Official Implementation:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311

        """
        (b, l, d) = x.shape

        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')
        x = F.silu(x)

        y = self.ssm(x)
        y = y * F.silu(res)

        output = self.out_proj(y)
        return output


    def ssm(self, x):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311

        """
        (d_in, n) = self.A_log.shape

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)

        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)

        y = self.selective_scan(x, delta, A, B, C, D)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]
        return y


    def selective_scan(self, u, delta, A, B, C, D):
        """Does selective scan algorithm. See:
            - Section 2 State Space Models in the Mamba paper [1]
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        This is the classic discrete state space formula:
            x(t + 1) = Ax(t) + Bu(t)
            y(t)     = Cx(t) + Du(t)
        except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).

        Args:
            u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
            delta: shape (b, l, d_in)
            A: shape (d_in, n)
            B: shape (b, l, n)
            C: shape (b, l, n)
            D: shape (d_in,)

        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
            Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.

        """
        (b, l, d_in) = u.shape
        n = A.shape[1]

        # Discretize continuous parameters (A, B)
        # - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper [1])
        # - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
        #   "A is the more important term and the performance doesn't change much with the simplification on B"
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

        # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
        #! Note that the below is sequential, while the official implementation does a much faster
        #! parallel scan that is additionally hardware-aware (like FlashAttention).
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):  #! get much slower for bigger l (= max_length K)
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)

        y = y + u * D
        return y


class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output
#********** ********* ********** **********


class Block(nn.Module):
    def __init__(self, config, index):  #, scale=False
        super().__init__()
        hidden_size = config.n_embd
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        if config.model_type == 'dmamba':
            self.norm_mamba = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
            self.mamba = Mamba(hidden_size)
        if config.model_type == 'dmamba-min':
            self.norm_mamba = RMSNorm(hidden_size)
            self.mamba = MambaBlock(ModelArgs(d_model=hidden_size))

        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp_channels = nn.Sequential(
            nn.Linear(hidden_size, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, hidden_size),
            nn.Dropout(config.drop_p),
        )

    def forward(self, x):
        x = x + self.mamba(self.norm_mamba(x))
        x = x + self.mlp_channels(self.ln_2(x))
        return x
