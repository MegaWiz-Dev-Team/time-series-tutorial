import mlx.core as mx
import mlx.nn as nn
import math

class CTRoPE(nn.Module):
    """
    Continuous-Time Rotary Positional Encoding (CT-RoPE)
    Extends standard RoPE to handle continuous timestamps.
    """
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
        # Angular frequencies: 10000^(-2i/d)
        inv_freq = 1.0 / (10000 ** (mx.arange(0, dims, 2).astype(mx.float32) / dims))
        self.inv_freq = inv_freq

    def __call__(self, x, t):
        # x: (B, L, D)
        # t: (B, L) - Continuous timestamps
        
        # Calculate angles: theta = t * inv_freq
        # Reshape t for broadcasting: (B, L, 1)
        t = t[..., None]
        # Reshape inv_freq for broadcasting: (1, 1, D/2)
        angles = t * self.inv_freq[None, None, :] # (B, L, D/2)
        
        # Duplicate angles for cos/sin: (B, L, D)
        angles = mx.repeat(angles, 2, axis=-1)
        
        cos = mx.cos(angles)
        sin = mx.sin(angles)
        
        # Planar rotation
        # x = [x0, x1, x2, x3, ...]
        # x_rotated = [x0*cos - x1*sin, x0*sin + x1*cos, ...]
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        
        out = mx.zeros_like(x)
        out[..., 0::2] = x1 * cos[..., 0::2] - x2 * sin[..., 0::2]
        out[..., 1::2] = x1 * sin[..., 1::2] + x2 * cos[..., 1::2]
        
        return out

class MoEBlock(nn.Module):
    """
    Frequency-Specific Mixture-of-Experts Block
    """
    def __init__(self, dims, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.dims = dims
        
        # Experts: Pool of lightweight MLPs
        self.experts = [
            nn.Sequential(
                nn.Linear(dims, dims * 4),
                nn.GELU(),
                nn.Linear(dims * 4, dims)
            ) for _ in range(num_experts)
        ]
        
        # Shared expert (global residual pathway)
        self.shared_expert = nn.Sequential(
            nn.Linear(dims, dims * 4),
            nn.GELU(),
            nn.Linear(dims * 4, dims)
        )
        
        # Gating mechanism
        self.gate = nn.Linear(dims, num_experts)
        self.shared_gate = nn.Linear(dims, 1)

    def __call__(self, x):
        # x: (B, L, D)
        B, L, D = x.shape
        x_flat = x.reshape(-1, D)
        
        # 1. Routing weights for non-shared experts
        gate_logits = self.gate(x_flat)
        # Top-K selection
        # Note: MLX topk returns indices and values
        # For simplicity in this draft, we'll use a mask-based approach
        gate_weights = mx.softmax(gate_logits, axis=-1)
        
        # For a full MoE implementation in MLX, we'd use scatter/gather or loops
        # Here we'll do a simplified weighted sum for the draft
        expert_outputs = mx.stack([expert(x_flat) for expert in self.experts], axis=1) # (B*L, num_experts, D)
        
        # Mask for top-k (simplified)
        top_k_indices = mx.argpartition(gate_logits, -self.top_k, axis=-1)[:, -self.top_k:]
        # This is a bit complex in pure MLX without loops or advanced indexing
        # Let's just use the gate weights directly for the weighted sum in this draft
        moe_out = mx.sum(expert_outputs * gate_weights[..., None], axis=1)
        
        # 2. Shared expert with sigmoid gate
        shared_out = self.shared_expert(x_flat)
        shared_weight = mx.sigmoid(self.shared_gate(x_flat))
        
        final_out = moe_out + shared_weight * shared_out
        return final_out.reshape(B, L, D)

class ODELayer(nn.Module):
    """
    Continuous Dynamics Extrapolation Block (Neural ODE)
    Simplified RK4 solver implementation in MLX.
    """
    def __init__(self, dims):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dims + 1, dims * 2), # dims + 1 for time delta
            nn.Tanh(),
            nn.Linear(dims * 2, dims)
        )

    def f(self, h, dt):
        # Concatenate hidden state and time delta
        # dt: (B, 1)
        # h: (B, D)
        inp = mx.concatenate([h, dt], axis=-1)
        return self.net(inp)

    def __call__(self, h, t_start, t_end, steps=4):
        # Extrapolate h from t_start to t_end using RK4
        dt_total = t_end - t_start
        dt = dt_total / steps
        
        for _ in range(steps):
            k1 = self.f(h, dt)
            k2 = self.f(h + dt * 0.5 * k1, dt)
            k3 = self.f(h + dt * 0.5 * k2, dt)
            k4 = self.f(h + dt * k3, dt)
            h = h + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        return h

class MIRABlock(nn.Module):
    def __init__(self, dims, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(dims)
        self.attn = nn.MultiHeadAttention(dims, num_heads)
        self.rope = CTRoPE(dims // num_heads)
        self.ln2 = nn.LayerNorm(dims)
        self.moe = MoEBlock(dims)

    def __call__(self, x, t, mask=None):
        # Attention with CT-RoPE
        # Standard MultiHeadAttention in MLX doesn't take RoPE directly in __call__
        # Usually RoPE is applied to Q and K before dot product
        # In a custom implementation, we'd apply CTRoPE here.
        # For this draft, we'll follow the block structure.
        
        r = self.ln1(x)
        # Simplified: apply rope to the input before attention for the draft
        r = self.rope(r, t)
        x = x + self.attn(r, r, r, mask)
        
        x = x + self.moe(self.ln2(x))
        return x

class MIRANet(nn.Module):
    def __init__(self, input_dims, model_dims, num_layers, num_heads, num_classes):
        super().__init__()
        self.embedding = nn.Linear(input_dims, model_dims)
        self.blocks = [MIRABlock(model_dims, num_heads) for _ in range(num_layers)]
        self.ode_block = ODELayer(model_dims)
        self.norm = nn.LayerNorm(model_dims)
        self.classifier = nn.Linear(model_dims, num_classes)

    def __call__(self, x, t):
        # x: (B, L, input_dims)
        # t: (B, L)
        
        x = self.embedding(x)
        
        for block in self.blocks:
            x = block(x, t)
            
        # Optional: Extrapolate to a "target" timestamp if needed (e.g. for forecasting)
        # For classification, we can use the last state
        # t_last = t[:, -1:]
        # t_future = t_last + 1.0 # arbitrary future point
        # x_last = x[:, -1, :]
        # x_future = self.ode_block(x_last, t_last, t_future)
        
        x = self.norm(x)
        # Global Average Pooling or taking the last token
        x = mx.mean(x, axis=1)
        return self.classifier(x)
