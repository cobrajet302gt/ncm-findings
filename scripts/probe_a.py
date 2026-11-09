#!/usr/bin/env python
# A1 codec ceiling / A2 level-ablation / A3 code-corruption sensitivity (xsum gold summaries)
import json, os, pickle, random, sys, time
import torch
from omegaconf import OmegaConf

REPO = "/home/cobrajet/ncm-lab/ncm"
sys.path.insert(0, REPO)
from tokenization.vqvae_tokenizer import VQVAETokenizer

CK = "/home/cobrajet/ncm-lab/artifacts/xsum"
DATA = "/home/cobrajet/ncm-lab/data/xsum"
N = int(os.environ.get("A_N", "512"))
random.seed(0); torch.manual_seed(0)

def norm(t):
    t = t.replace("<|endoftext|>", " ").replace("<|pad|>", " ")
    return " ".join(t.lower().replace("\n", " ").split())

def toks(t):
    return norm(t).split()

def rouge_l(ref, hyp):
    a, b = toks(ref), toks(hyp)
    if not a or not b:
        return 0.0, 0.0
    m, prev = 0, [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b):
            cur[j + 1] = prev[j] + 1 if x == y else max(cur[j], prev[j + 1])
        prev = cur
    lcs = prev[len(b)]
    p, r = lcs / len(b), lcs / len(a)
    f1 = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    inter = len(set(a) & set(b))
    r1 = 0.0 if not a else inter / len(a)
    return f1, r1

CFG = OmegaConf.load(os.path.join(REPO, "config/eval/xsum.yaml"))
CFG.tokenizer.vqvae.checkpoint_path = os.path.join(CK, CFG.checkpoints.tokenizer)
tk = VQVAETokenizer.from_config(CFG.tokenizer)
tk.vqvae = tk.vqvae.cuda().eval()

# device-safe replacements built on the canonical encode->flat codes->decode path
# (repo's single-text reconstruct() is dead code: vqvae.reconstruct unpacks vq() wrong)
def _codes(text):
    idx = torch.tensor(tk._adjust_length(tk.pre_tokenizer.encode(text)), dtype=torch.long, device="cuda")
    return tk.encode_pre_tokenized_idx(idx)  # (B, sum(levels))
def _recon(text):
    return tk.decode(_codes(text)[0].tolist())
def _recon_ms(text):
    return tk.decode_multiscales(_codes(text))[0]
tk.reconstruct = _recon
tk.reconstruct_multiscales = _recon_ms
K = len(CFG.tokenizer.vqvae.vector_quantizer_config.levels.manual_levels)
VCAB = CFG.tokenizer.vqvae.vqvae_vocab_size[0]
print(f"levels K={K} seg_len={CFG.tokenizer.vqvae.context_length} vcab={VCAB}", flush=True)

with open(os.path.join(DATA, "reference_summaries.pickle"), "rb") as f:
    refs = pickle.load(f)
refs = [r if isinstance(r, str) else " ".join(map(str, r)) for r in refs[:N]]
print(f"{len(refs)} gold summaries loaded", flush=True)

def acc_rouge(pairs):
    rl = sum(rouge_l(r, h)[0] for r, h in pairs) / len(pairs)
    r1 = sum(rouge_l(r, h)[1] for r, h in pairs) / len(pairs)
    ex = sum(norm(r) == norm(h) for r, h in pairs) / len(pairs)
    return {"exact": round(ex, 4), "rougeL": round(rl, 4), "rouge1": round(r1, 4)}

out = {"n": len(refs), "K": K, "A1_codec_ceiling": {}, "A2_level_curve": [], "A3_corruption": {}}

with torch.no_grad():
    t0 = time.time()
    # A1: full-code round trip (perfect codes -> decode)
    pairs = [(s, tk.reconstruct(s)) for s in refs]
    out["A1_codec_ceiling"] = acc_rouge(pairs)
    print("A1", out["A1_codec_ceiling"], f"{time.time()-t0:.0f}s", flush=True)

    # A2: quality per cumulative level (coarse -> fine)
    per_level = [[] for _ in range(K)]
    for s in refs:
        multis = tk.reconstruct_multiscales(s)
        assert len(multis) == K, f"got {len(multis)} levels"
        for k, hyp in enumerate(multis):
            per_level[k].append((s, hyp if isinstance(hyp, str) else str(hyp)))
    for k, pairs in enumerate(per_level):
        out["A2_level_curve"].append(acc_rouge(pairs))
        print("A2 level", k + 1, out["A2_level_curve"][-1], f"{time.time()-t0:.0f}s", flush=True)

    # A3: sensitivity to wrong conditioning codes (uniform replacement at rate rho)
    for rho in (0.1, 0.3, 0.5, 0.7):
        pairs = []
        for s in refs:
            ids = _codes(s)[0].tolist()
            corrupt = [random.randrange(VCAB) if random.random() < rho else c for c in ids]
            idx = torch.tensor([corrupt], dtype=torch.long, device="cuda")
            hyp = tk.decode_multiscales(idx, skip_special_tokens=True)[0][-1]  # final-level decode of corrupted codes
            pairs.append((s, hyp if isinstance(hyp, str) else str(hyp)))
        out["A3_corruption"][str(rho)] = acc_rouge(pairs)
        print("A3 rho", rho, out["A3_corruption"][str(rho)], f"{time.time()-t0:.0f}s", flush=True)

os.makedirs("/home/cobrajet/ncm-lab/eval", exist_ok=True)
json.dump(out, open("/home/cobrajet/ncm-lab/eval/A1A2A3-xsum.json", "w"), indent=1)
print("DONE", flush=True)
