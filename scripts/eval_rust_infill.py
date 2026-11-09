"""B1c-v2 — coarse-plan conditioning eval (the B9 thesis, measured).

The architecture's conditioning interface IS the level-0 code (prefix_len=1),
so "spec supplies the coarse plan" = give the TRUE L0 code of the held-out
snippet instead of letting the model draw one from the corpus prior. Paired:
same snippets, N samples each arm, same temp. Verifier = tree-sitter parse.
If true-L0 >> prior-L0, external coarse plans are the lever and B9 is real.
"""
import os, sys, json, glob, random, time
import numpy as np
sys.path.insert(0, "/home/cobrajet/ncm-lab/ncm")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "6")))
from omegaconf import OmegaConf
from models.next_concept import NextConceptTransformer
from tokenization.vqvae_tokenizer import VQVAETokenizer
import tree_sitter_rust as tsr
import tree_sitter as ts

LAB = "/home/cobrajet/ncm-lab"
CK = os.environ.get("CK", f"{LAB}/artifacts/rustcodec-4k")
CK_NAME = os.environ.get("CK_NAME", "tokenizer_2500.pt")
GEN = os.environ.get("GEN", f"{LAB}/train/rust3")
N = int(os.environ.get("N", "8"))
NSNIP = int(os.environ.get("NSNIP", "64"))
B = int(os.environ.get("BATCH", "8"))
GEN_TEMP = float(os.environ.get("GEN_TEMP", "1.0"))
TOPK = int(os.environ.get("TOP_K", "0"))
torch.manual_seed(0)
DEV = torch.device(os.environ.get("DEV", "cpu"))

CFG = OmegaConf.load(os.path.join(CK, "config.yaml"))
CFG.tokenizer.vqvae.checkpoint_path = os.path.join(CK, CK_NAME)
CFG.tokenizer.pre_tokenizer.vocab_file_path = os.path.join(CK, "bpe_vocab_8192.pickle")
CFG.tokenizer.pre_tokenizer.merges_file_path = os.path.join(CK, "bpe_merges_8192.pickle")
tk = VQVAETokenizer.from_config(CFG.tokenizer)
tk.vqvae = tk.vqvae.to(DEV).eval()
for p in tk.vqvae.parameters(): p.requires_grad_(False)

class NS: pass
sd = torch.load(os.path.join(GEN, "gen_last.pt"), map_location="cpu", weights_only=False)
c = NS()
for k, v in sd["cfg"].items(): setattr(c, k, v)
net = NextConceptTransformer(vqvae_local=tk.vqvae, config=c).to(DEV)
net.load_state_dict(sd["model"], strict=True)
net.eval()
P = int(CFG.model.prefix_len)
EOS = tk.pre_tokenizer.eos_token_id
PAD = tk.pre_tokenizer.pad_token_id
LEVELS = list(tk.vqvae.vq.levels)
SEG = int(CFG.tokenizer.vqvae.context_length)
V0 = int(CFG.tokenizer.vqvae.vqvae_vocab_size[0])
parser = ts.Parser(ts.Language(tsr.language()))
uni = torch.tensor(json.load(open(f"{CK}/l0_unigram-tokenizer_2500.json")), dtype=torch.float, device=DEV)

# held-out items (same regex + zero-overlap exclusion as codec eval)
import re
CORPUS_NAMES = set()
for d in sorted(glob.glob(f"{LAB}/data/rust/corpus/*/")):
    CORPUS_NAMES.add(d.split("/")[-2])
files = [f for f in glob.glob("/home/cobrajet/strata/**/*.rs", recursive=True)
         if "target" not in f and not any(n in f.split("/") for n in CORPUS_NAMES)]
snips = []
for f in files:
    t = open(f, encoding="utf-8", errors="ignore").read()
    for m in re.finditer(r"(?:pub |)(?:async fn|fn|macro_rules!|impl|struct|enum|trait)\s[\s\S]{0,800}?\n\}", t):
        s = m.group(0)
        if 120 < len(s) < 900:
            snips.append(s)
random.Random(1).shuffle(snips)
snips = snips[:NSNIP]
print(f"cond {len(snips)} snippets x {N} samples x 2 arms; P={P} SEG={SEG}", flush=True)

def true_l0(sn):
    ids = tk.pre_tokenizer.encode(sn)[:SEG]
    x = torch.tensor([ids + [PAD] * (SEG - len(ids))], dtype=torch.long, device=DEV)
    codes = tk.encode_pre_tokenized_idx(x)
    return int(codes[0, 0])

def gen_batch(l0_col, b, seed):
    with torch.no_grad():
        l0 = l0_col.to(DEV)
        pfx = torch.full((b, P), EOS, dtype=torch.long, device=DEV)
        g = net.generate(l0, pfx, temperature=GEN_TEMP, top_k=TOPK, top_p=1.0)
        _, zl = tk.vqvae.vq.idx_to_zhats(g.to(DEV), return_zlevels=True)
        ids = tk.vqvae.decoder(zl[-1]).argmax(-1)
    out = []
    for row in ids.tolist():
        if EOS in row: row = row[:row.index(EOS)]
        out.append(tk.pre_tokenizer.decode([x for x in row if x not in (EOS, PAD)], skip_special_tokens=True))
    return out

def decode_multiscale(codes_row):
    parts = tk.decode_multiscales([int(x) for x in codes_row[: sum(LEVELS)]])
    return "".join(str(t).replace("<.>", " ") for t in parts)

def decode_multiscale(codes_row):
    parts = tk.decode_multiscales(codes_row)
    return "".join(str(t).replace("<.>", " ") for t in parts)

rows = []
t0 = time.time()
for si, sn in enumerate(snips):
    tl0 = true_l0(sn)
    stat = {s: {"parse": 0, "n": 0} for s in ("true_l0", "prior_l0")}
    for s0 in range(0, N, B):
        b = min(B, N - s0)
        gen = torch.Generator(device=DEV.type).manual_seed(si * 100 + s0)
        l0t = torch.full((b, 1), tl0, dtype=torch.long)
        l0p = torch.multinomial(uni.expand(b, V0), 1, generator=gen).to(torch.long)
        for arm, l0col in (("true_l0", l0t), ("prior_l0", l0p)):
            for txt in gen_batch(l0col, b, si * 100 + s0):
                if not txt.strip(): continue
                stat[arm]["n"] += 1
                stat[arm]["parse"] += int(not parser.parse(txt.encode("utf-8", "ignore")).root_node.has_error)
    rows.append({"snip": si, "l0": tl0, **stat})
    print(f"  snip {si}: true_l0 {stat['true_l0']['parse']}/{stat['true_l0']['n']} | prior {stat['prior_l0']['parse']}/{stat['prior_l0']['n']}", flush=True)

def agg(k):
    n = sum(r[k]["n"] for r in rows); p = sum(r[k]["parse"] for r in rows)
    anyp = sum(1 for r in rows if r[k]["parse"] > 0)
    return {"parse_rate": round(p / max(1, n), 4), "any_parse_rate": round(anyp / max(1, len(rows)), 4), "n": n}
res = {"gen": os.path.basename(GEN), "n_snips": len(rows), "temp": GEN_TEMP,
       "true_l0": agg("true_l0"), "prior_l0": agg("prior_l0"),
       "l0_used_distinct": {"true": len({r["l0"] for r in rows}), "snips": len(rows)},
       "secs": round(time.time() - t0), "per_snippet": rows}
json.dump(res, open(f"{LAB}/eval/rust-infill-c1.json", "w"), indent=1)
print("B1c", json.dumps({k: v for k, v in res.items() if k != "per_snippet"}))
print("DONE")
