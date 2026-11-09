#!/usr/bin/env python
# Stage-2 generator trainer, mirroring next_concept.generate() level-for-level with
# teacher-forced CORRUPTED conditioning codes (paper Alg 1). Frozen official TinyStories codec.
# Sized to train alongside the live judge in GPU1's ~1.9 GB slice.
import json, os, pickle, sys, time
import torch, torch.nn.functional as F
from omegaconf import OmegaConf

REPO = "/home/cobrajet/ncm-lab/ncm"; sys.path.insert(0, REPO)
from tokenization.vqvae_tokenizer import VQVAETokenizer
from models.next_concept import NextConceptTransformer

CK = "/home/cobrajet/ncm-lab/artifacts/tinystories"
STEPS = int(os.environ.get("TRAIN_STEPS", "60"))
BS = int(os.environ.get("BS", "32"))
NU = int(os.environ.get("NU", "384")); LAYERS = int(os.environ.get("LAYERS", "6")); HEADS = int(os.environ.get("HEADS", "6"))
LR = float(os.environ.get("LR", "3e-4"))
C0, C1 = float(os.environ.get("CORR0", "0.5")), float(os.environ.get("CORR1", "0.25"))
OUT = os.environ.get("OUT", "/home/cobrajet/ncm-lab/train/smoke")
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0)

CFG = OmegaConf.load(os.path.join(REPO, "config/eval/tinystories.yaml"))
CFG.tokenizer.vqvae.checkpoint_path = os.path.join(CK, CFG.checkpoints.tokenizer)
CFG.tokenizer.pre_tokenizer.vocab_file_path = os.path.join(CK, CFG.checkpoints.bpe_vocab)
CFG.tokenizer.pre_tokenizer.merges_file_path = os.path.join(CK, CFG.checkpoints.bpe_merges)
tk = VQVAETokenizer.from_config(CFG.tokenizer)
tk.vqvae = tk.vqvae.cuda().eval()
for p in tk.vqvae.parameters():
    p.requires_grad_(False)

class NS: pass
MC = OmegaConf.to_container(CFG.model, resolve=True); MC.update(n_layer=LAYERS, n_embd=NU, n_head=HEADS, dropout=0.1)
cfgns = NS()
for k, v in MC.items(): setattr(cfgns, k, v)
net = NextConceptTransformer(vqvae_local=tk.vqvae, config=cfgns).cuda()
print("generator params:", f"{sum(p.numel() for p in net.parameters() if p.requires_grad)/1e6:.1f}M", flush=True)

rows = pickle.load(open("/home/cobrajet/ncm-lab/data/tinystories/train_bpe_8192.pickle", "rb"))
rows = [r for r in rows if 8 <= len(r) <= 120]
print("usable train rows:", len(rows), flush=True)
CTX = int(CFG.tokenizer.vqvae.context_length)
vq = tk.vqvae.vq
LEVELS = list(vq.levels)
K = len(LEVELS)
VC = int(CFG.tokenizer.vqvae.vqvae_vocab_size[0])
EOS = tk.pre_tokenizer.eos_token_id
P = int(CFG.model.prefix_len)
LVL_FIRST = getattr(net, "_lvl_first", 1)
QUERY_L0 = bool(getattr(net, "query_level0", True))
dev = torch.device("cuda")

def batch(idx):
    seg = [tk._adjust_length(list(rows[i])) for i in idx]
    seg = [s[0] for s in seg if len(s[0]) == CTX]
    return torch.tensor(seg, dtype=torch.long, device=dev) if seg else None

def true_codes(xseg):
    with torch.no_grad():
        flat = tk.encode_pre_tokenized_idx(xseg)  # (B, sum diffs)=128 cumulative layout
    return flat

def split_levels(flat):
    # generate() outputs cat(per-level FULL blocks); flat layout = exclusive cumsum offsets
    off, o = [], 0
    for L in LEVELS:
        off.append(o); o += L
    return [flat[:, off[k]:off[k] + LEVELS[k]] for k in range(K)]
opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.1)
RESUME = os.environ.get("RESUME", "")
step0 = 0
if RESUME and os.path.exists(os.path.join(RESUME, "gen_last.pt")):
    sd = torch.load(os.path.join(RESUME, "gen_last.pt"), map_location="cpu", weights_only=False)
    net.load_state_dict(sd["model"]); step0 = int(sd.get("steps_done", sd.get("steps", 0)))
    hist = list(sd.get("hist", []))
    print("resumed from", RESUME, "at step", step0, flush=True)
oom_skips = 0

def save_ckpt(steps_done):
    torch.save({"model": {k: v.cpu() for k, v in net.state_dict().items()}, "hist": hist,
                "cfg": MC, "steps_done": steps_done, "bs": BS}, os.path.join(OUT, "gen_last.pt"))
    json.dump({"hist": hist, "steps_done": steps_done, "oom_skips": oom_skips},
              open(os.path.join(OUT, "train.json"), "w"), indent=1)

def prefix_emb_of(B, dtype):
    pfx = torch.full((B, P), EOS, dtype=torch.long, device=dev)
    z_prefix = tk.vqvae.encoder.wte(pfx)[:, :P, :]
    return net.transformer.wte(z_prefix).to(dtype).detach()  # 1-token prefix: detach so per-level backwards stay independent

t0 = time.time()
hist = list(locals().get("hist", []))
for step in range(step0, STEPS):
    g = torch.Generator().manual_seed(step * 7919 + int(os.environ.get("SEED", "0")))
    idx = torch.randint(0, len(rows), (BS * 2,), generator=g).tolist()[:BS]
    xseg = batch(idx)
    if xseg is None or xseg.shape[0] < BS // 2: continue
    B = xseg.shape[0]
    rho = C0 + (C1 - C0) * min(1.0, step / max(1, STEPS - 1))
    flat = true_codes(xseg)
    rtil = torch.where(torch.rand_like(flat, dtype=torch.float) < rho,
                       torch.randint(0, VC, flat.shape, device=dev), flat)
    # per-level FULL cumulative slices (generate() feeds embeddings of level-k codes where
    # idx_next holds that level's L_k flat-layout slice)
    def slice_level(arr, k):
        off, o = 0, []
        for L in LEVELS:
            o.append(off); off += L
        return arr[:, o[k]:o[k] + LEVELS[k]]
    opt.zero_grad(set_to_none=True)
    lsum = 0.0
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pfx = prefix_emb_of(B, torch.bfloat16)
        # level 0 via query path (non-AR branch of generate())
        if QUERY_L0:
            l0 = LEVELS[0]
            q = net.build_l0_query_input(B, torch.bfloat16)
            q = net.transformer.wte(q)
            if not net.use_rope: q = q + net.wpe(net.wp_1L[:, :l0])
            if net.config.use_level_encoding:
                _w0 = net.transformer.wle(net.lvl_1L[:, :l0])
                q = q + _w0[:, :l0]
            h = torch.cat([pfx, q], dim=1)
            m0 = torch.zeros(1, 1, P + l0, P + l0, device=dev); m0[:, :, :P, P:] = float("-inf")
            rope0 = (net.rope_cos_gen_0, net.rope_sin_gen_0) if net.use_rope else None
            for blk in net.transformer.blocks: h = blk(h, attn_mask=m0, rope_cos_sin=rope0)
            h = net.transformer.ln_f(h)
            lg0 = net.lm_heads[0](h[:, P:P + l0, :])
            t0c = slice_level(flat, 0)
            ce = F.cross_entropy(lg0.reshape(-1, lg0.shape[-1]).float(), t0c.reshape(-1))
            lsum += float(ce); (ce / K).backward()  # per-level backward: independent graphs, frozen conditioning
        if not net.config.use_level_encoding:
            level_pos_embed = torch.zeros(1, net.L - LEVELS[0], net.config.n_embd, device=dev)
        z_hat = torch.zeros(B, LEVELS[-1], tk.vqvae.embed_dim, device=dev)
        level_acum = LEVELS[0] if QUERY_L0 else 0  # verbatim bookkeeping from generate()
        for k in range(K - 1):
            r_k = slice_level(rtil, k)
            z_q_level = vq.embeddings[k](r_k)
            z_hat, x = vq.get_next_autoregressive_input(k, z_hat, z_q_level)
            if net.use_post_upsample_conv:
                x = net._apply_post_upsample_conv_single(x, k + (1 - LVL_FIRST))
            if net.config.use_level_encoding:
                # per-level SEPARATE wle forward on its own slice of the (1, L) level-index buffer
                lpk = net.transformer.wle(net.lvl_1L[:, level_acum:level_acum + LEVELS[k + 1]])
                x = net.transformer.wte(x) + lpk[:, :x.shape[1], :]
            else:
                x = net.transformer.wte(x) + level_pos_embed[:, level_acum:level_acum + LEVELS[k + 1], :]
            hidx = k + (1 - LVL_FIRST)
            logits = net._decode_level_logits(x, pfx, k, None, hidx=hidx)
            tgt = slice_level(flat, k + 1)
            ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1))
            lsum += float(ce); (ce / K).backward()
            del logits, x, ce
            level_acum += LEVELS[k + 1]
    opt.step()
    hist.append(round(lsum / K, 4))
    if (step + 1) % 500 == 0:
        save_ckpt(step + 1); torch.cuda.empty_cache()
    if step % 250 == 0 or step == STEPS - 1:
        print(f"step {step} rho {rho:.2f} loss {hist[-1]} elapsed {time.time()-t0:.0f}s vram {torch.cuda.max_memory_allocated()/2**20:.0f}MiB", flush=True)

torch.save({"model": {k: v.cpu() for k, v in net.state_dict().items()}, "hist": hist, "cfg": MC, "steps": STEPS, "bs": BS},
           os.path.join(OUT, "gen_last.pt"))
json.dump({"hist": hist, "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20,
           "secs_per_step": (time.time() - t0) / max(1, STEPS)}, open(os.path.join(OUT, "train.json"), "w"), indent=1)
print("DONE", flush=True)
