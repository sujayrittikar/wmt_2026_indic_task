"""CA-front-end byte-level decoder-only language model.

Standalone, importable copy of the architecture defined in `04_ca_pretrain.ipynb` — class names and
submodule attribute names are kept **identical** so a checkpoint saved by the notebook loads cleanly
here. This file is also uploaded alongside the weights so the Hugging Face repo is self-contained.

A vocabulary-free UTF-8-byte model: byte+position embedding → a **causal Neural-CA front-end** (one
weight-shared local update rule iterated K steps, left-looking so a next-byte LM has no future leakage)
→ a standard pre-norm causal Transformer → weight-tied byte head.

    from ca_byte_lm import load_pretrained, generate
    model, cfg, meta = load_pretrained("runs/ca/nca/best.pt", device="cuda")
    print(generate(model, cfg, meta["lang_tag"]["Khasi"], meta, device="cuda"))
"""
from types import SimpleNamespace
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- byte vocabulary specials (must match the notebook) ----
PAD, BOS, EOS = 256, 257, 258


class CausalNCA(nn.Module):
    """One weight-shared local update rule, iterated K steps, left-looking (causal)."""
    def __init__(self, d, hidden, steps, kernel, fire_rate):
        super().__init__()
        self.steps, self.fire_rate, self.pad = steps, fire_rate, kernel - 1
        self.norm = nn.LayerNorm(d)
        self.perceive = nn.Conv1d(d, d, kernel_size=kernel, groups=d, bias=False)  # depthwise, causal via left pad
        self.fc1 = nn.Linear(2 * d, hidden)
        self.fc2 = nn.Linear(hidden, d)
        nn.init.zeros_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)              # start = identity
    def forward(self, s):
        for _ in range(self.steps):
            sn = self.norm(s)
            p = F.pad(sn.transpose(1, 2), (self.pad, 0))          # left-pad only -> no future leakage
            p = self.perceive(p).transpose(1, 2)
            delta = self.fc2(F.gelu(self.fc1(torch.cat([sn, p], dim=-1))))
            if self.training and self.fire_rate < 1.0:
                mask = (torch.rand(s.shape[:2], device=s.device) <= self.fire_rate).unsqueeze(-1)
                delta = delta * mask
            s = s + delta
        return s


class CausalConvStack(nn.Module):
    """Control: K DISTINCT causal-conv blocks (no weight sharing) -> ~K x the NCA params, same depth."""
    def __init__(self, d, hidden, steps, kernel, **_):
        super().__init__()
        self.pad = kernel - 1
        self.blocks = nn.ModuleList()
        for _ in range(steps):
            self.blocks.append(nn.ModuleDict(dict(
                norm=nn.LayerNorm(d),
                perceive=nn.Conv1d(d, d, kernel_size=kernel, groups=d, bias=False),
                fc1=nn.Linear(2 * d, hidden), fc2=nn.Linear(hidden, d))))
        for b in self.blocks:
            nn.init.zeros_(b["fc2"].weight); nn.init.zeros_(b["fc2"].bias)
    def forward(self, s):
        for b in self.blocks:
            sn = b["norm"](s)
            p = F.pad(sn.transpose(1, 2), (self.pad, 0))
            p = b["perceive"](p).transpose(1, 2)
            s = s + b["fc2"](F.gelu(b["fc1"](torch.cat([sn, p], dim=-1))))
        return s


def build_frontend(cfg):
    if cfg.FRONTEND == "nca":
        return CausalNCA(cfg.d_model, cfg.nca_hidden, cfg.nca_steps, cfg.nca_kernel, cfg.nca_fire_rate)
    if cfg.FRONTEND == "conv":
        return CausalConvStack(cfg.d_model, cfg.nca_hidden, cfg.nca_steps, cfg.nca_kernel)
    return nn.Identity()


class Block(nn.Module):
    def __init__(self, d, nh, dff, dropout):
        super().__init__()
        self.nh, self.hd = nh, d // nh
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d); self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, dff), nn.GELU(), nn.Linear(dff, d), nn.Dropout(dropout))
        self.drop = nn.Dropout(dropout); self.ad = dropout
    def forward(self, x, attn_mask=None):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q = q.view(B, T, self.nh, self.hd).transpose(1, 2)
        k = k.view(B, T, self.nh, self.hd).transpose(1, 2)
        v = v.view(B, T, self.nh, self.hd).transpose(1, 2)
        dp = self.ad if self.training else 0.0
        if attn_mask is None:                                   # default: pure causal (pretraining/LM)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dp)
        else:                                                   # explicit mask (e.g. prefix-LM for MT)
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dp)
        a = a.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.drop(self.proj(a))
        x = x + self.mlp(self.ln2(x))
        return x


class CAByteLM(nn.Module):
    def __init__(self, cfg, vocab):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(vocab, cfg.d_model)
        self.pos = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.frontend = build_frontend(cfg)
        self.blocks = nn.ModuleList([Block(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
                                     for _ in range(cfg.n_layers)])
        self.lnf = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, vocab, bias=False)
        self.head.weight = self.tok.weight                          # weight tying
        self.apply(self._init)
        if isinstance(self.frontend, CausalNCA):
            nn.init.zeros_(self.frontend.fc2.weight); nn.init.zeros_(self.frontend.fc2.bias)
        elif isinstance(self.frontend, CausalConvStack):
            for b in self.frontend.blocks:
                nn.init.zeros_(b["fc2"].weight); nn.init.zeros_(b["fc2"].bias)
    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
    def forward(self, idx, targets=None, attn_mask=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok(idx) + self.pos(pos))
        x = self.frontend(x)                                        # causal -> no leakage
        for b in self.blocks: x = b(x, attn_mask)
        logits = self.head(self.lnf(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                                   ignore_index=PAD)
        return logits, loss


# ───────────────────────── helpers (encode / load / generate) ─────────────────────────
def encode(text, tag_id):
    """[lang tag] + utf-8 bytes + [EOS] — the notebook's exact sentence encoding."""
    return [tag_id] + list(text.encode("utf-8")) + [EOS]


def cfg_from_dict(d):
    """Rebuild a config namespace from the dict stored in the checkpoint."""
    return SimpleNamespace(**d)


def prefix_lm_mask(prefix_lens, T, device):
    """[B,1,T,T] bool attention mask for prefix-LM translation: positions inside the source prefix
    attend bidirectionally *within the prefix*; everything else is causal. (True = attend.)"""
    ar = torch.arange(T, device=device)
    causal = ar[None, :] <= ar[:, None]                         # [T,T]  j <= i
    P = torch.as_tensor(prefix_lens, device=device).view(-1, 1) # [B,1]
    pref_i = ar[None, :] < P                                    # [B,T]  position is in the prefix
    pref = pref_i[:, :, None] & pref_i[:, None, :]              # [B,T,T] both i and j in prefix
    return (causal[None] | pref).unsqueeze(1)                   # [B,1,T,T]


def load_pretrained(ckpt_path, device="cpu"):
    """Return (model, cfg, meta) from a notebook/HF checkpoint. meta has vocab, lang_tag, step, bpb."""
    ck = torch.load(ckpt_path, map_location=device)
    cfg = cfg_from_dict(ck["cfg"])
    model = CAByteLM(cfg, ck["vocab"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    meta = {k: ck[k] for k in ("vocab", "lang_tag", "step", "bpb") if k in ck}
    return model, cfg, meta


def from_hub(repo_id, filename="ca_byte_lm.pt", device="cpu"):
    """Download the checkpoint from the Hugging Face Hub and load it. Returns (model, cfg, meta)."""
    from huggingface_hub import hf_hub_download
    return load_pretrained(hf_hub_download(repo_id, filename), device=device)


@torch.no_grad()
def translate(model, cfg, meta, src_text, src_lang, tgt_lang,
              max_new=256, no_repeat_ngram=8, prefix_lm=True, device="cpu"):
    """Translate one sentence with a *tuned* CA checkpoint (notebook 05), by prompted continuation:
    `[src_lang] src_bytes [tgt_lang]` -> generate target bytes to EOS. Greedy + no-repeat-ngram, with
    the prefix-LM mask (source attends bidirectionally) that the translator was tuned with.

        model, cfg, meta = from_hub("user/ca-byte-mt-en2x", device="cuda")
        translate(model, cfg, meta, "The government announced a new policy.",
                  "English", "Khasi", device="cuda")
    """
    model.eval()
    if isinstance(model.frontend, CausalNCA):
        model.frontend.fire_rate = 1.0
    TAG, SEQ = meta["lang_tag"], cfg.seq_len
    prompt = ([TAG[src_lang]] + list(str(src_text).encode("utf-8")) + [TAG[tgt_lang]])[-SEQ:]
    plen, ids, gen = len(prompt), list(prompt), []
    for _ in range(max_new):
        ctx = torch.tensor([ids[-SEQ:]], device=device)
        mask = prefix_lm_mask([min(plen, ctx.size(1))], ctx.size(1), device) if prefix_lm else None
        logits, _ = model(ctx, attn_mask=mask)
        lg = logits[0, -1].float()
        if no_repeat_ngram and len(gen) >= no_repeat_ngram - 1:        # block repeated byte n-grams
            pref = tuple(gen[-(no_repeat_ngram - 1):])
            for i in range(len(gen) - no_repeat_ngram + 1):
                if tuple(gen[i:i + no_repeat_ngram - 1]) == pref:
                    lg[gen[i + no_repeat_ngram - 1]] = -1e9
        nxt = int(lg.argmax())
        if nxt == EOS:
            break
        gen.append(nxt); ids.append(nxt)
    return bytes([t for t in gen if t < 256]).decode("utf-8", errors="replace")


@torch.no_grad()
def generate(model, cfg, tag_id, meta=None, n_bytes=160, temp=0.8, top_k=40, device="cpu"):
    """Sample a monolingual continuation conditioned on a language tag id."""
    model.eval()
    if isinstance(model.frontend, CausalNCA):
        model.frontend.fire_rate = 1.0
    idx = torch.tensor([[tag_id]], device=device)
    for _ in range(n_bytes):
        ctx = idx[:, -cfg.seq_len:]
        logits, _ = model(ctx)
        logits = logits[:, -1, :].float() / temp
        if top_k:
            v, _ = torch.topk(logits, top_k); logits[logits < v[:, [-1]]] = -float("inf")
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
        if nxt.item() == EOS:
            break
        idx = torch.cat([idx, nxt], dim=1)
    return bytes([t for t in idx[0].tolist() if t < 256]).decode("utf-8", errors="replace")
