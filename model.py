import math

import torch
import torch.nn.functional as F
from torch import nn
from collections import defaultdict

class Rotary(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


def rmsnorm(x0, eps=1e-6):
    x = x0.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.type_as(x0)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        
    def forward(self, x):
        x = x + self.attn(rmsnorm(x))
        x = x + self.mlp(rmsnorm(x))
        return x


class GPTConfig:
    def __init__(self, vocab_size=50257, n_layer=12, n_head=12, n_embd=768):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd

        self.emb_std = 0.02
        self.base_std = 0.02


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # init wte with emb_std
        self.transformer.wte.weight.data.normal_(mean=0.0, std=config.emb_std)
        # init lm_head with zeros
        self.lm_head.weight.data.fill_(0)

        # init all weights with fan_in / 1024 * base_std
        for n, p in self.named_parameters():
            skip_list = ['wte', 'lm_head']
            if not any([s in n for s in skip_list]):
                fan_in = p.shape[1]

                p.data.normal_(mean=0.0, std=config.base_std * (1024 / fan_in) ** 0.5)


    def forward(self, idx, targets=None, return_logits=True):
        b, t = idx.size()

        x = self.transformer.wte(idx)

        for block in self.transformer.h:
            x = block(x)
        x = rmsnorm(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        if not return_logits:
            logits = None

        return logits, loss

    def configure_classic_optimizers(self, weight_decay, learning_rate, betas, device_type):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=betas
        )
        return optimizer

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        no_decay_name_list = ["bias", "norm"]

        optimizer_grouped_parameters = []
        final_optimizer_settings = {}

        param_groups = defaultdict(lambda: {"params": [], "weight_decay": None, "lr": None})

        for n, p in self.named_parameters():
            if p.requires_grad:
                
                # Define learning rate for specific types of params
                if any(ndnl in n for ndnl in no_decay_name_list):
                    lr_value = learning_rate * 0.033
                    weight_decay_value = 0.0
                else:
                    hidden_dim = p.shape[-1]
                    lr_value = learning_rate * (32 / hidden_dim)
                    weight_decay_value = 0.1 * hidden_dim / 1024 # weight decay 0.1 (SP: 1024)
                
                # in the case of embedding layer, we use higher lr.
                if "wte" in n:
                    lr_value = learning_rate * 0.1
                    weight_decay_value = 0.0

                group_key = (lr_value, weight_decay_value)
                param_groups[group_key]["params"].append(p)
                param_groups[group_key]["weight_decay"] = weight_decay_value
                param_groups[group_key]["lr"] = lr_value

                final_optimizer_settings[n] = {
                    "lr": lr_value,
                    "wd": weight_decay_value,
                    "shape": str(list(p.shape)),
                }

        optimizer_grouped_parameters = [v for v in param_groups.values()]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, betas=betas)

        return optimizer