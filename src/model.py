import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# rns norm cell from tutorial
class RMSNorm(nn.Module):
    """
    Root mean square layer normalisation.
    params:
        d:   size of the dimension to normalise (the last dimension of the input)
        eps: constant added inside the square root for numerical stability
    """

    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(d))     # learned gain g, initialised to 1

    def forward(self, x):
        """
        params:
            x: (..., d)
        returns:
            tensor of the same shape as x
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)                     # upcast to avoid overflow when squaring

        # Step 1: root mean square over the LAST dimension, keeping the dimension for broadcasting
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)   # (..., 1)

        # Step 2: divide by the RMS and apply the learned gain
        out = (x / rms) * self.gain.to(torch.float32)                      # (..., d)

        return out.to(in_dtype)                     # cast back to the original dtype

# tutorial cell 
class SwiGLU(nn.Module):
    """
    Position-wise SwiGLU feed-forward network (no bias terms).
    params:
        d_model: model dimension (input and output size)
        d_ff:    inner hidden dimension, canonically about (8/3) * d_model
    """

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)      # gate branch
        self.w3 = nn.Linear(d_model, d_ff, bias=False)      # value branch
        self.w2 = nn.Linear(d_ff, d_model, bias=False)      # projection back to d_model

    def forward(self, x):
        """
        params:
            x: (..., d_model)
        returns:
            tensor of shape (..., d_model)
        """
        # SiLU(W1 x) is the gate; it multiplies W3 x element-wise; W2 projects back down.
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    
# add reul ffn class 
class ReLUFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()

        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w2(F.relu(self.w1(x)))

def compute_d_ff(d_model, multiple_of=64):
    """Round (8/3) * d_model up to the nearest multiple of `multiple_of`."""
    d_ff = int(8 * d_model / 3)
    return multiple_of * ((d_ff + multiple_of - 1) // multiple_of)


# tutorial cell rpe
class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary position embeddings (RoPE). Has no learnable parameters.
    params:
        d_head:      dimension of the query/key vectors (must be even)
        max_seq_len: longest sequence we will ever need positions for
        theta:       the base Theta in theta_k = Theta^(-2k/d)
    """

    def __init__(self, d_head, max_seq_len, theta=10000.0):
        super().__init__()
        assert d_head % 2 == 0, "RoPE rotates pairs of dimensions, so d_head must be even"

        k = torch.arange(0, d_head // 2, dtype=torch.float32)        # (d_head/2,)
        inv_freq = theta ** (-2.0 * k / d_head)                      # theta_k
        positions = torch.arange(max_seq_len, dtype=torch.float32)   # (max_seq_len,)
        angles = torch.outer(positions, inv_freq)                    # (max_seq_len, d_head/2)

        # Buffers, not parameters: these are fixed, and not saved into checkpoints.
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x, positions):
        """
        params:
            x:         (..., seq_len, d_head)
            positions: (seq_len,) absolute position of each element along the sequence axis
        returns:
            rotated tensor of the same shape as x
        """
        # Step 1: look up the precomputed angles for these positions
        cos = self.cos[positions]                   # (seq_len, d_head/2)
        sin = self.sin[positions]                   # (seq_len, d_head/2)

        # Step 2: split the last dimension into the even and odd members of each pair
        x_even, x_odd = x[..., 0::2], x[..., 1::2]  # each (..., seq_len, d_head/2)

        # Step 3: apply the 2D rotation to every pair
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos

        # Step 4: interleave the pairs back into the original layout
        return torch.stack((out_even, out_odd), dim=-1).flatten(-2)

# tutorial cell scaled dot product attention
def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Scaled dot-product attention.
    params:
        q:    (..., n_queries, d_k)
        k:    (..., n_keys, d_k)
        v:    (..., n_keys, d_v)
        mask: optional boolean (n_queries, n_keys); True = attend, False = do not attend
    returns:
        out:  (..., n_queries, d_v)
    """
    d_k = q.size(-1)

    # Step 1: scores e_ij = q_i . k_j / sqrt(d_k)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)        # (..., n_queries, n_keys)

    # Step 2: block the disallowed positions with -inf so softmax gives them zero weight
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    # Step 3: normalise over the KEY dimension, then take the weighted average of values
    attn = F.softmax(scores, dim=-1)                         # (..., n_queries, n_keys)
    return attn @ v                                          # (..., n_queries, d_v)

# tutorial cell causal attention
class CausalSelfAttention(nn.Module):
    """
    Causal multi-head self-attention with QK norm and RoPE.
    params:
        d_model:     model dimension
        n_heads:     number of attention heads
        rope:        a shared RotaryPositionalEmbedding module
        use_qk_norm: whether to RMSNorm the queries and keys before attention
    """

    def __init__(self, d_model, n_heads, rope, use_qk_norm=True, use_rope = True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads          # d_q = d_k = d_v = d_model / n_heads

        # One full-width projection each; the heads are carved out by reshaping.
        # No bias terms, following modern LLMs.
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.rope = rope
        self.use_rope = use_rope
        self.q_norm = RMSNorm(self.d_head) if use_qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.d_head) if use_qk_norm else nn.Identity()

    def forward(self, x):
        """
        params:
            x: (batch, seq_len, d_model)
        returns:
            (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device)

        # Step 1: project, then split into heads -> (batch, n_heads, seq_len, d_head)
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # Step 2: QK norm, then RoPE on queries and keys only (never on values)
        q, k = self.q_norm(q), self.k_norm(k)
        # make RoPe conditional
        if self.use_rope:
            q=self.rope(q, positions)
            k = self.rope(k, positions)

        # Step 3: causal mask - query i attends to key j only if j <= i
        mask = positions[None, :] <= positions[:, None]              # (seq_len, seq_len)

        out = scaled_dot_product_attention(q, k, v, mask)            # (b, n_heads, seq_len, d_head)

        # Step 4: concatenate the heads and project back to the residual stream
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(out)

# trnasformer block from the tutorial
class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block: attention sub-layer then feed-forward sub-layer.
    params:
        d_model, n_heads, d_ff: sizes
        rope:        shared RotaryPositionalEmbedding module
        use_qk_norm: passed through to the attention module
    """

    def __init__(self, d_model, n_heads, d_ff, rope, use_qk_norm=True, use_rmsnorm=True, use_rope=True, ffn_type="swiglu"):
        super().__init__()
        self.attn_norm = RMSNorm(d_model) if use_rmsnorm else nn.Identity()
        self.attn = CausalSelfAttention(d_model, n_heads, rope, use_qk_norm and use_rmsnorm, use_rope,)
        self.ffn_norm = RMSNorm(d_model) if use_rmsnorm else nn.Identity()
        # self.ffn = SwiGLU(d_model, d_ff)
        if ffn_type == "swiglu":
            self.ffn = SwiGLU(d_model, d_ff)
        elif ffn_type == "relu":
            self.ffn = ReLUFFN(d_model, 4 * d_model)
        else:
            raise ValueError(f"Unknown FFN type: {ffn_type}")

    def forward(self, x):
        """
        params:
            x: (batch, seq_len, d_model)
        returns:
            (batch, seq_len, d_model)
        """
        # Sub-layer 1: normalise, attend, add the residual.
        x = x + self.attn(self.attn_norm(x))
        # Sub-layer 2: normalise, feed-forward, add the residual.
        x = x + self.ffn(self.ffn_norm(x))
        return x

# cell from tutorial
# update from skeleton for 4.1 updating the config 
@dataclass
class TransformerConfig:
    vocab_size: int = 4000 # choice from 3.4
    context_length: int = 256      # maximum sequence length
    n_layers: int = 4
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 1344                # ~ (8/3) * d_model, rounded to a multiple of 64
    rope_theta: float = 10000.0
    use_qk_norm: bool = True    

    # additional config options for 4.1:
    use_rmsnorm: bool=True
    use_rope: bool=True
    ffn_type: str = "swiglu"
    
# full transformer lm fromt he tutorial
class TransformerLM(nn.Module):
    """
    A decoder-only Transformer language model in the modern dense style.
    params:
        config: a TransformerConfig
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model)

        # One RoPE module shared by every layer: the cos/sin tables are identical everywhere,
        # so there is no reason to store them more than once.
        self.rope = RotaryPositionalEmbedding(
            d_head=config.d_model // config.n_heads,
            max_seq_len=config.context_length,
            theta=config.rope_theta,
        )

        self.layers = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff,self.rope, config.use_qk_norm, config.use_rmsnorm, config.use_rope, config.ffn_type)
            for _ in range(config.n_layers)
        ])

        self.final_norm = RMSNorm(config.d_model) if config.use_rmsnorm else nn.Identity()
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            std = math.sqrt(2.0 / (module.in_features + module.out_features))
            nn.init.trunc_normal_(module.weight, std=std, a=-3 * std, b=3 * std)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids):
        """
        params:
            token_ids: (batch, seq_len) integer token IDs
        returns:
            logits: (batch, seq_len, vocab_size)
        """
        # Step 1: embed the token IDs. No positional embedding - RoPE handles position.
        x = self.token_embeddings(token_ids)              # (batch, seq_len, d_model)

        # Step 2: run the stack of pre-norm blocks
        for layer in self.layers:
            x = layer(x)                                  # (batch, seq_len, d_model)

        # Step 3: final norm, then project to vocabulary logits
        return self.lm_head(self.final_norm(x))           # (batch, seq_len, vocab_size)

    def num_parameters(self, non_embedding=False):
        """Total parameter count; optionally excluding the embedding and LM head."""
        total = sum(p.numel() for p in self.parameters())
        if non_embedding:
            total -= self.token_embeddings.weight.numel() + self.lm_head.weight.numel()
        return total