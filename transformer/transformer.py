# http://peterbloem.nl/blog/transformers
import torch
from torch import nn
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):
    def __init__(self, k, heads=8):
        super().__init__()
        self.k, self.heads = k, heads

        # Weight matrices for keys, queries and values.
        self.w_k = nn.Linear(k, k * heads, bias=False)
        self.w_q = nn.Linear(k, k * heads, bias=False)
        self.w_v = nn.Linear(k, k * heads, bias=False)

        # Unifies the heads into a single k-dim vector.
        self.unifyheads = nn.Linear(heads * k, k)

    def forward(self, x):
        b, t, k = x.size()
        h = self.heads

        queries = self.w_q(x).view(b, t, h, k)
        keys = self.w_k(x).view(b, t, h, k)
        values = self.w_v(x).view(b, t, h, k)

        # Fold heads into batch dimension.
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, k)
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, k)
        values = values.transpose(1, 2).contiguous().view(b * h, t, k)

        # Scaling here should save some memory.
        queries = queries / (k ** (1 / 4))
        keys = keys / (k ** (1 / 4))

        weights = torch.bmm(queries, keys.transpose(1, 2))  # Shape (b*h, t, t)
        weights = F.softmax(weights, dim=2)
        out = torch.bmm(weights, values).view(b, h, t, k)

        out = out.transpose(1, 2).contiguous().view(b, t, h * k)
        return self.unifyheads(out)

    def forward_einsum(self, x):
        """
        Alternative formulation with Einstein notation.
        """
        b, t, k = x.size()
        h = self.heads

        queries = self.w_q(x).view(b, t, h, k)
        keys = self.w_k(x).view(b, t, h, k)
        values = self.w_v(x).view(b, t, h, k)

        # w_ti = q^T_t k_i
        weights = torch.einsum("bthk,bihk->bhti", queries, keys) / math.sqrt(k)
        weights = F.softmax(weights, dim=-1)
        # y_i = w_ij v_j
        out = torch.einsum("bhtd,bdhk->bthk", weights, values)

        self.unifyheads.weight.view(k, h, k)  # Can be moved to init.
        out = torch.einsum("bthd,dhk->btk", out, self.unifyheads)
        return out + self.unifyheads.bias


class TransformerBlock(nn.Module):
    def __init__(self, k, heads):
        super().__init__()

        self.attention = SelfAttention(k, heads=heads)

        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)

        hidden_dim = 4 * k
        self.ff = nn.Sequential(
            nn.Linear(k, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, k)
        )

    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended + x)

        fedforward = self.ff(x)
        return self.norm2(fedforward + x)


class Transformer(nn.Module):
    def __init__(self, k, heads, depth, seq_length, num_tokens, num_classes):
        super().__init__()

        self.num_tokens = num_tokens
        self.token_emb = nn.Embedding(num_tokens, k)
        self.pos_emb = nn.Embedding(seq_length, k)

        tblocks = []
        for i in range(depth):
            tblocks.append(TransformerBlock(k=k, heads=heads))
        self.tblocks = nn.Sequential(*tblocks)

        self.to_probs = nn.Linear(k, num_classes)

    def forward(self, x):
        """
        x: (b, t) tensor of integer values representing words (in some vocabulary).
        return: (b, c) tensor of log-probabilites over the classes.
        """
        tokens = self.token_emb(x)
        b, t, k = tokens.size()

        positions = torch.arange(t)
        positions = self.pos_emb(positions)[None, :, :].expand(b, t, k)

        x = tokens + positions
        x = self.tblocks(x)

        x = self.toprobs(x.mean(dim=1))
        return F.log_softmax(x, dim=1)
