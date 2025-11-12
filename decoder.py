# ==========================================================
# ✅ DLCV HW3 Part 2 — Final Correct Decoder (LoRA version + robust generate)
# ----------------------------------------------------------
# ✔ 與 train_decoder.py / inference.py 直接對接
# ✔ 支援以 inputs_embeds 起步（視覺前綴 + prompt）或 input_ids 起步
# ✔ LoRA 僅訓練 Q/K/V/O，參數 < 10M（不改動現有結構）
# ✔ 內建 temperature / top_p / repetition_penalty / eos 停止
# ==========================================================

from typing import Optional
import torch
from torch import nn
import loralib as lora

# ==========================================================
# 基本設定
# ==========================================================
class Config:
    def __init__(self):
        self.vocab_size = 151936
        self.hidden_size = 1024
        self.intermediate_size = 3072
        self.num_hidden_layers = 28
        self.num_attention_heads = 16
        self.num_key_value_heads = 8
        self.head_dim = 128
        self.max_position_embeddings = 40960
        self.rms_norm_eps = 1e-6
        self.rope_theta = 1000000
        self.rope_scaling = None


# ==========================================================
# RMSNorm
# ==========================================================
class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# ==========================================================
# MLP
# ==========================================================
class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.functional.silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ==========================================================
# Rotary Embedding 工具
# ==========================================================
def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos + rotate_half(q) * sin), (k * cos + rotate_half(k) * sin)


def compute_default_rope_parameters(config: Optional[Config] = None, device=None):
    base = config.rope_theta
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dim = int(head_dim)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    return inv_freq, 1.0


# ==========================================================
# Attention
# ==========================================================
def repeat_kv(hidden_states: torch.Tensor, n_rep: int):
    b, n_kv, s, d = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(b, n_kv, n_rep, s, d)
    return hidden_states.reshape(b, n_kv * n_rep, s, d)


def eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    attn_weights = attn_weights.masked_fill(attention_mask, float("-inf"))
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
    return attn_output, attn_weights


class Qwen3Attention(nn.Module):
    def __init__(self, config: Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5

        # ✅ LoRA on Q/K/V/O（保留，不改）
        self.q_proj = lora.Linear(config.hidden_size, config.num_attention_heads * self.head_dim,
                                  r=16, lora_alpha=32, lora_dropout=0.05, merge_weights=False, bias=False)
        self.k_proj = lora.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim,
                                  r=16, lora_alpha=32, lora_dropout=0.05, merge_weights=False, bias=False)
        self.v_proj = lora.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim,
                                  r=16, lora_alpha=32, lora_dropout=0.05, merge_weights=False, bias=False)
        self.o_proj = lora.Linear(config.num_attention_heads * self.head_dim, config.hidden_size,
                                  r=16, lora_alpha=32, lora_dropout=0.05, merge_weights=False, bias=False)

        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, attention_mask):
        bsz, seq_len, _ = hidden_states.size()
        shape = (bsz, seq_len, -1, self.head_dim)
        q = self.q_norm(self.q_proj(hidden_states).view(shape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states).view(shape)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(shape).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        attn_output, _ = eager_attention_forward(self, q, k, v, attention_mask, scaling=self.scaling)
        attn_output = attn_output.reshape(bsz, seq_len, -1)
        return self.o_proj(attn_output), None


# ==========================================================
# Decoder Layer
# ==========================================================
class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Config, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3Attention(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, attention_mask):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states, position_embeddings, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


# ==========================================================
# Rotary Embedding
# ==========================================================
class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, config: Config, device=None):
        super().__init__()
        self.config = config
        inv_freq, scale = compute_default_rope_parameters(config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.scale = scale

    def forward(self, x, position_ids):
        inv_freq = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        pos = position_ids[:, None, :].float()
        freqs = (inv_freq @ pos).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos() * self.scale, emb.sin() * self.scale


# ==========================================================
# Decoder 主體
# ==========================================================
class Decoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=151643)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.config = config

    # -------- 主要前向：支援 inputs_embeds 或 input_ids --------
    def forward(self, input_ids=None, inputs_embeds=None, position_ids=None, attention_mask=None):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        h = inputs_embeds
        if position_ids is None:
            position_ids = torch.arange(0, h.size(1), device=h.device).unsqueeze(0)

        pos_emb = self.rotary_emb(h, position_ids)

        if attention_mask is None:
            L = h.size(1)
            mask = torch.triu(torch.ones(L, L, device=h.device, dtype=torch.bool), diagonal=1)
            attention_mask = mask.unsqueeze(0).unsqueeze(0)

        for layer in self.layers:
            h = layer(h, pos_emb, attention_mask)

        h = self.norm(h)
        return self.lm_head(h)

    # -------- 取樣工具 --------
    @staticmethod
    def _apply_repetition_penalty(logits, generated_ids, penalty: float):
        if generated_ids is None or penalty == 1.0:
            return logits
        # 將已生成過的 token logits 降權（ >1 代表懲罰 ）
        uniq = generated_ids.unique()
        logits.index_copy_(dim=-1, index=uniq, source=logits[..., uniq] / penalty)
        return logits

    @staticmethod
    def _top_p_filter(logits, top_p: float):
        if top_p >= 1.0:
            return logits
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cum = torch.cumsum(sorted_probs, dim=-1)
        mask = cum > top_p
        # 保留第一個超過 top_p 的位置
        mask[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(mask, 0.0)
        # 轉回 logits（加一個小常數避免 log(0)）
        new_logits = torch.full_like(logits, float("-inf"))
        new_logits.scatter_(dim=-1, index=sorted_idx, src=torch.log(sorted_probs + 1e-12))
        return new_logits

    # -------- 生成：支援從 inputs_embeds 或 input_ids 起步 --------
    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        max_new_tokens: int = 30,
        eos_token_id: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
    ):
        device = next(self.parameters()).device

        # 起點：embedding 或 id
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("generate() 需要 input_ids 或 inputs_embeds 其中之一。")
            input_ids = input_ids.to(device)
            cur_embeds = self.embed_tokens(input_ids)
        else:
            cur_embeds = inputs_embeds.to(device)
            if input_ids is None:
                # 若沒有提供對應的 input_ids，建立空的 ids，用於追蹤生成序列
                input_ids = torch.empty((cur_embeds.size(0), 0), dtype=torch.long, device=device)

        B = cur_embeds.size(0)
        generated = input_ids  # 追蹤完整 token 序列（只包含文字 token）

        for _ in range(max_new_tokens):
            logits = self(inputs_embeds=cur_embeds)[:, -1, :]  # 只取最後一步
            # 重複懲罰
            logits = self._apply_repetition_penalty(logits, generated, repetition_penalty)

            # 溫度/Top-p
            if temperature != 1.0:
                logits = logits / max(temperature, 1e-6)
            logits = self._top_p_filter(logits, top_p)

            # 取樣（或貪婪當作特例）
            if top_p < 1.0 or temperature != 1.0:
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)

            # 續接 ids 與 embeds
            generated = torch.cat([generated, next_id], dim=1)
            next_emb = self.embed_tokens(next_id)  # (B,1,H)
            cur_embeds = torch.cat([cur_embeds, next_emb], dim=1)

            # 早停
            if eos_token_id is not None:
                # 若 batch 中所有樣本都生成到 eos，則停止
                if torch.all(next_id.squeeze(-1) == eos_token_id):
                    break

        return generated
