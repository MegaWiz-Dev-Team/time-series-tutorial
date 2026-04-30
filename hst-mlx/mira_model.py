import mlx.core as mx
import mlx.nn as nn
import math


class CTRoPE(nn.Module):
    """
    Continuous-Time Rotary Positional Encoding.
    Applied to Q and K inside RoPEMultiHeadAttention, not to the full embedding.
    """
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
        inv_freq = 1.0 / (10000 ** (mx.arange(0, dims, 2).astype(mx.float32) / dims))
        self.inv_freq = inv_freq

    def __call__(self, x, t):
        # x: (B, L, D), t: (B, L)
        t = t[..., None]                                    # (B, L, 1)
        angles = t * self.inv_freq[None, None, :]           # (B, L, D/2)
        angles = mx.repeat(angles, 2, axis=-1)              # (B, L, D)
        cos = mx.cos(angles)
        sin = mx.sin(angles)
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        out = mx.zeros_like(x)
        out[..., 0::2] = x1 * cos[..., 0::2] - x2 * sin[..., 0::2]
        out[..., 1::2] = x1 * sin[..., 1::2] + x2 * cos[..., 1::2]
        return out


class RoPEMultiHeadAttention(nn.Module):
    """
    Multi-head attention with CT-RoPE applied correctly to Q and K only.
    """
    def __init__(self, dims, num_heads):
        super().__init__()
        assert dims % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dims // num_heads
        self.q_proj = nn.Linear(dims, dims, bias=False)
        self.k_proj = nn.Linear(dims, dims, bias=False)
        self.v_proj = nn.Linear(dims, dims, bias=False)
        self.out_proj = nn.Linear(dims, dims, bias=False)
        self.rope = CTRoPE(self.head_dim)
        self.scale = math.sqrt(self.head_dim)

    def __call__(self, x, t, mask=None):
        # x: (B, L, D), t: (B, L)
        B, L, D = x.shape
        H, Dh = self.num_heads, self.head_dim

        # Project and reshape to (B, H, L, Dh)
        Q = self.q_proj(x).reshape(B, L, H, Dh).transpose(0, 2, 1, 3)
        K = self.k_proj(x).reshape(B, L, H, Dh).transpose(0, 2, 1, 3)
        V = self.v_proj(x).reshape(B, L, H, Dh).transpose(0, 2, 1, 3)

        # Apply CTRoPE to Q and K per-head: flatten to (B*H, L, Dh)
        Q = Q.reshape(B * H, L, Dh)
        K = K.reshape(B * H, L, Dh)
        # Tile timestamps across heads: (B, L) -> (B*H, L)
        t_tiled = mx.tile(t[:, None, :], (1, H, 1)).reshape(B * H, L)
        Q = self.rope(Q, t_tiled)
        K = self.rope(K, t_tiled)
        Q = Q.reshape(B, H, L, Dh)
        K = K.reshape(B, H, L, Dh)

        # Scaled dot-product attention
        scores = mx.matmul(Q, K.transpose(0, 1, 3, 2)) / self.scale  # (B, H, L, L)
        if mask is not None:
            scores = scores + mask
        attn = mx.softmax(scores, axis=-1)
        out = mx.matmul(attn, V)                            # (B, H, L, Dh)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, D)
        return self.out_proj(out)


class MoEBlock(nn.Module):
    """
    Frequency-Specific Mixture-of-Experts with true sparse top-k routing.
    Uses double-argsort rank masking to select top-k experts per token.
    """
    def __init__(self, dims, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = [
            nn.Sequential(nn.Linear(dims, dims * 4), nn.GELU(), nn.Linear(dims * 4, dims))
            for _ in range(num_experts)
        ]
        self.shared_expert = nn.Sequential(
            nn.Linear(dims, dims * 4), nn.GELU(), nn.Linear(dims * 4, dims)
        )
        self.gate = nn.Linear(dims, num_experts)
        self.shared_gate = nn.Linear(dims, 1)

    def __call__(self, x):
        B, L, D = x.shape
        x_flat = x.reshape(-1, D)                           # (N, D)

        gate_logits = self.gate(x_flat)                     # (N, E)

        # Sparse top-k: rank each expert per token, keep only top-k
        # double argsort gives rank in descending order
        ranks = mx.argsort(mx.argsort(-gate_logits, axis=-1), axis=-1)
        top_k_mask = ranks < self.top_k                     # (N, E) True for top-k
        masked_logits = mx.where(top_k_mask, gate_logits, mx.full(gate_logits.shape, float('-inf')))
        gate_weights = mx.softmax(masked_logits, axis=-1)   # (N, E) sparse

        expert_outputs = mx.stack([e(x_flat) for e in self.experts], axis=1)  # (N, E, D)
        moe_out = mx.sum(expert_outputs * gate_weights[..., None], axis=1)    # (N, D)

        shared_out = self.shared_expert(x_flat)
        shared_weight = mx.sigmoid(self.shared_gate(x_flat))
        return (moe_out + shared_weight * shared_out).reshape(B, L, D)


class ODELayer(nn.Module):
    """
    Continuous dynamics extrapolation via RK4.
    t_start and t_end must be (B, 1) tensors so dt is properly batched.
    """
    def __init__(self, dims):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dims + 1, dims * 2),
            nn.Tanh(),
            nn.Linear(dims * 2, dims),
        )

    def f(self, h, dt):
        # h: (B, D), dt: (B, 1)
        return self.net(mx.concatenate([h, dt], axis=-1))

    def __call__(self, h, t_start, t_end, steps=4):
        # t_start, t_end: (B, 1) — keeps dt shape (B, 1) for concat with h (B, D)
        dt = (t_end - t_start) / steps
        for _ in range(steps):
            k1 = self.f(h, dt)
            k2 = self.f(h + 0.5 * dt * k1, dt)
            k3 = self.f(h + 0.5 * dt * k2, dt)
            k4 = self.f(h + dt * k3, dt)
            h = h + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return h


class MIRABlock(nn.Module):
    def __init__(self, dims, num_heads, num_experts=8, top_k=2):
        super().__init__()
        self.ln1 = nn.LayerNorm(dims)
        self.attn = RoPEMultiHeadAttention(dims, num_heads)
        self.ln2 = nn.LayerNorm(dims)
        self.moe = MoEBlock(dims, num_experts, top_k)

    def __call__(self, x, t, mask=None):
        x = x + self.attn(self.ln1(x), t, mask)
        x = x + self.moe(self.ln2(x))
        return x


class MIRANet(nn.Module):
    def __init__(self, input_dims, model_dims, num_layers, num_heads,
                 num_experts=8, top_k=2, num_classes=5):
        super().__init__()
        self.embedding = nn.Linear(input_dims, model_dims)
        self.blocks = [MIRABlock(model_dims, num_heads, num_experts, top_k)
                       for _ in range(num_layers)]
        self.ode_block = ODELayer(model_dims)
        self.norm = nn.LayerNorm(model_dims)
        self.classifier = nn.Linear(model_dims, num_classes)

    def __call__(self, x, t):
        # x: (B, L, input_dims), t: (B, L)
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x, t)
        x = self.norm(x)

        # Global average + ODE extrapolation of last token to next window boundary
        x_avg = mx.mean(x, axis=1)          # (B, D)
        x_last = x[:, -1, :]               # (B, D)
        t_last = t[:, -1:]                 # (B, 1)
        t_future = t_last + 1.0
        x_future = self.ode_block(x_last, t_last, t_future)

        return self.classifier(x_avg + x_future)
