#!/usr/bin/env python
# Rust-B1: sample from the trained Rust generator (train/rust1) and gate the output on
# (a) tree-sitter grammar validity (exact parse, no rustc), (b) distinct-4 diversity,
# (c) eyeball examples. Level-0 plan sampled from the MODEL'S OWN prior (lg0 logits),
# not uniform randint — the NCM-zero lesson built in.
import json, os, sys, time
import torch, torch.nn.functional as F
from omegaconf import OmegaConf
sys.path.insert(0, "/home/cobrajet/ncm-lab/ncm")
import tree_sitter_rust as tsr
import tree_sitter as ts
from tokenization.vqvae_tokenizer import VQVAETokenizer
from models.next_concept import NextConceptTransformer

LAB = "/home/cobrajet/ncm-lab"
CK = os.environ.get("CK", f"{LAB}/artifacts/rustcodec")
GEN = os.environ.get("GEN", f"{LAB}/train/rust1")
N = int(os.environ.get("N", "128")); B = int(os.environ.get("B", "32"))
TEMP = float(os.environ.get("GEN_TEMP", "1.0")); TOPK = int(os.environ.get("TOP_K", "0"))  # NB: never os.environ["TEMP"] — set system-wide to TMPDIR here
torch.manual_seed(0)
DEV = torch.device(os.environ.get("DEV", "cuda"))

CFG = OmegaConf.load(os.path.join(CK, "config.yaml"))
CK_NAME = os.environ.get("CK_NAME", "tokenizer.pt")
CFG.tokenizer.vqvae.checkpoint_path = os.path.join(CK, CK_NAME)
CFG.tokenizer.pre_tokenizer.vocab_file_path = os.path.join(CK, "bpe_vocab_8192.pickle")
CFG.tokenizer.pre_tokenizer.merges_file_path = os.path.join(CK, "bpe_merges_8192.pickle")
tk = VQVAETokenizer.from_config(CFG.tokenizer)
tk.vqvae = tk.vqvae.to(DEV).eval()
for p in tk.vqvae.parameters(): p.requires_grad_(False)

class NS: pass
sd = torch.load(os.path.join(GEN, "gen_last.pt"), map_location="cpu", weights_only=False)
MC = sd["cfg"]
c = NS()
for k, v in MC.items(): setattr(c, k, v)
net = NextConceptTransformer(vqvae_local=tk.vqvae, config=c).to(DEV)
net.load_state_dict(sd["model"], strict=True)
net.eval()
print("gen step:", sd.get("steps_done", sd.get("steps")), flush=True)

P = int(CFG.model.prefix_len)
EOS = tk.pre_tokenizer.eos_token_id
PAD = tk.pre_tokenizer.pad_token_id
LEVELS = list(tk.vqvae.vq.levels)
V0 = int(CFG.tokenizer.vqvae.vqvae_vocab_size[0])
parser = ts.Parser(ts.Language(tsr.language()))

def sample_rows(logits_row, gen):
    x = logits_row.float() / max(TEMP, 1e-5)
    if TOPK: 
        kth = torch.topk(x, TOPK, dim=-1).values[..., -1:]
        x = x.masked_fill(x < kth, float("-inf"))
    return torch.multinomial(F.softmax(x, -1), 1, generator=gen)

def l0_model_prior(Bsz, gen):
    # only valid when config query_level0=True (this model: False)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pfx = torch.full((Bsz, P), EOS, dtype=torch.long, device=DEV)
        z = tk.vqvae.encoder.wte(pfx)[:, :P, :]
        pfx = net.transformer.wte(z).to(torch.bfloat16)
        l0 = LEVELS[0]
        q = net.build_l0_query_input(Bsz, torch.bfloat16)
        q = net.transformer.wte(q)
        if not net.use_rope: q = q + net.wpe(net.wp_1L[:, :l0])
        if net.config.use_level_encoding:
            q = q + net.transformer.wle(net.lvl_1L[:, :l0])[:, :l0]
        h = torch.cat([pfx, q], dim=1)
        m0 = torch.zeros(1, 1, P + l0, P + l0, device=DEV); m0[:, :, :P, P:] = float("-inf")
        rope0 = (net.rope_cos_gen_0, net.rope_sin_gen_0) if net.use_rope else None
        for blk in net.transformer.blocks: h = blk(h, attn_mask=m0, rope_cos_sin=rope0)
        h = net.transformer.ln_f(h)
        lg0 = net.lm_heads[0](h[:, P + l0 - 1:P + l0, :])
        return sample_rows(lg0[:, 0, :], gen).to(torch.long)

# query_level0=False (this arch): level-0 is a pure INPUT -> sample from EMPIRICAL corpus
# unigram, the exact trick official checkpoints ship as inference.level0_unigram.
V0 = int(CFG.tokenizer.vqvae.vqvae_vocab_size[0])
UNIF = f"{CK}/l0_unigram-{CK_NAME.replace('.pt', '')}.json"
if os.path.exists(UNIF):
    uni = torch.tensor(json.load(open(UNIF)), dtype=torch.float, device=DEV)
else:
    segs = torch.load(f"{CK}/segments.pt", map_location="cpu", weights_only=True)
    counts = torch.zeros(V0)
    g2 = torch.Generator().manual_seed(1)
    idx = torch.randint(0, segs.shape[0], (4096,), generator=g2)
    with torch.no_grad():
        for s0 in range(0, 4096, 512):
            xb = segs[idx[s0:s0 + 512]].to(DEV)
            flat = tk.encode_pre_tokenized_idx(xb)[:, :LEVELS[0]]
            for c in flat.flatten().tolist():
                counts[c] += 1
    uni = torch.tensor(counts, dtype=torch.float, device=DEV)
    json.dump(counts.tolist(), open(UNIF, "w"))
    print("l0 empirical prior built:", UNIF, "top8", counts.topk(8).indices.tolist(), flush=True)

def l0_prior(Bsz, gen):
    if getattr(net, "query_level0", False):
        return l0_model_prior(Bsz, gen)
    return torch.multinomial(uni.expand(Bsz, V0), 1, generator=gen).to(torch.long)

codes_all, texts = [], []
gen = torch.Generator(device=DEV.type).manual_seed(0)
t0 = time.time()
for s0 in range(0, N, B):
    b = min(B, N - s0)
    l0 = l0_prior(b, gen).to(DEV)
    codes_all.append(l0.cpu())
    pfx = torch.full((b, P), EOS, dtype=torch.long, device=DEV)  # generate() wants TOKEN ids; it does its own wte
    with torch.no_grad():
        g = net.generate(l0, pfx, temperature=TEMP, top_k=TOPK, top_p=1.0)
        _, zl = tk.vqvae.vq.idx_to_zhats(g.to(DEV), return_zlevels=True)
        ids = tk.vqvae.decoder(zl[-1]).argmax(-1)
    for row in ids.tolist():
        if EOS in row: row = row[:row.index(EOS)]
        t = tk.pre_tokenizer.decode([x for x in row if x not in (EOS, PAD)], skip_special_tokens=True)
        texts.append(t)
    print(f"sampled {len(texts)} ({time.time()-t0:.0f}s)", flush=True)

parse_ok = sum(1 for t in texts if not parser.parse(t.encode("utf-8", "ignore")).root_node.has_error)
def distinct4(ts_):
    tot, uniq = 0, set()
    for t in ts_:
        w = t.split()
        for i in range(len(w) - 3):
            tot += 1; uniq.add(tuple(w[i:i+4]))
    return (len(uniq)/tot) if tot else 0.0
l0c = torch.cat(codes_all).flatten().tolist()
res = dict(n=len(texts), temp=TEMP, top_k=TOPK, parse_ok=parse_ok,
           parse_rate=round(parse_ok/len(texts), 4), distinct4=round(distinct4(texts), 4),
           mean_len_chars=round(sum(len(t) for t in texts)/len(texts), 1),
           l0_prior_hist=[l0c.count(v) for v in range(V0)],
           secs=round(time.time()-t0))
print("B1", json.dumps(res), flush=True)
import random
for t in random.Random(0).sample(texts, min(5, len(texts))):
    print("----\n" + t[:400], flush=True)
json.dump({"res": res, "examples": texts[:12]}, open(f"{LAB}/eval/rust-b1-{os.path.basename(GEN)}.json", "w"), indent=1)
