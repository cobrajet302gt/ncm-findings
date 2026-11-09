#!/usr/bin/env python
# Held-out verdict eval for the Rust codec: strata's own .rs files (zero overlap with the
# training corpus) -> BPE -> 128-token blocks -> codec round-trip -> text-exact %, block-
# exact %, token-acc %. The number that decides the Rust lane. CPU or GPU via DEV.
import glob, json, os, re, sys, time
import torch
sys.path.insert(0, "/home/cobrajet/ncm-lab/ncm")
from models.vqvae import VQVAE
from tokenization.bpe_tokenizer import BPETokenizer
from omegaconf import OmegaConf

LAB = "/home/cobrajet/ncm-lab"
CK = f"{LAB}/artifacts/rustcodec"
DEV = os.environ.get("DEV", "cuda")
BATCH = int(os.environ.get("BATCH", "64"))
SEG = 128
torch.manual_seed(0)

class Cfg:
    vocab_file_path, merges_file_path = f"{CK}/bpe_vocab_8192.pickle", f"{CK}/bpe_merges_8192.pickle"
    special_tokens = {"eos_token": "<|endoftext|>", "pad_token": "<|pad|>"}
bpe = BPETokenizer.from_config(Cfg())
_rv = bpe.reverse_vocab
EOS, PAD = _rv[b"<|endoftext|>"], _rv[b"<|pad|>"]
VOCAB = bpe.vocab_size

cfg = OmegaConf.load(f"{CK}/config.yaml")
tvq = OmegaConf.to_container(cfg.tokenizer.vqvae, resolve=True)
vqvae = VQVAE(in_vocab_size=VOCAB, context_length=SEG, embed_dim=tvq["embed_dim"],
              vqvae_vocab_size=tvq["vqvae_vocab_size"], n_layers=tvq["n_layers"],
              vector_quantizer_config=cfg.tokenizer.vqvae.vector_quantizer_config,
              compression_factor=tvq["compression_factor"], dropout=0.0,
              pre_quant_groupnorm=tvq["pre_quant_groupnorm"], use_qk_norm=tvq["use_qk_norm"],
              pre_quant_dropout=0.0)
sd = torch.load(f"{CK}/tokenizer.pt", map_location="cpu", weights_only=False)
vqvae.load_state_dict(sd["model"], strict=True)
vqvae = vqvae.eval().to(DEV)

# held-out snippets: strata repo only
files = [f for f in glob.glob("/home/cobrajet/strata/**/*.rs", recursive=True) if "target" not in f]
snips = []
for f in files:
    t = open(f, encoding="utf-8", errors="ignore").read()
    for m in re.finditer(r"(?:pub |)(?:async fn|fn|macro_rules!|impl|struct|enum|trait)\s[\s\S]{0,800}?\n\}", t):
        s = m.group(0)
        if 120 < len(s) < 900:
            snips.append(s)
print(f"held-out: {len(files)} strata files, {len(snips)} snippets", flush=True)

def blocks(ids):
    return [ids[i:i + SEG] for i in range(0, len(ids), SEG)]

tot_b = tok_ok = chunk_ok = tot_tok = 0
tot_s = snip_exact = 0
err_loc = {"comment": 0, "string": 0, "punct": 0, "ident": 0}
worst = []
t0 = time.time()
with torch.no_grad():
    for s0 in range(0, len(snips), BATCH):
        batch_sn = snips[s0:s0 + BATCH]
        flat, owner, lens = [], [], []
        for si, sn in enumerate(batch_sn):
            ids = bpe.encode(sn)
            bl = blocks(ids)
            for b in bl:
                flat.append(b + [PAD] * (SEG - len(b))); owner.append(si); lens.append(len(b))
            tot_s += 0
        x = torch.tensor(flat, dtype=torch.long, device=DEV)
        idx = vqvae.encode(x)
        logits = vqvae.decode(idx)
        pred = logits.argmax(-1)
        recs = [""] * len(batch_sn)
        pos = {}
        for k in range(len(flat)):
            si = owner[k]; ln = lens[k]
            row = pred[k, :ln].tolist()
            piece = bpe.decode(row)
            recs[si] += piece
            ctx = ""
            for j, (a, b_) in enumerate(zip(pred[k, :ln].tolist(), flat[k][:ln])):
                if a != b_:
                    tb = bpe.vocab.get(b_, b"").decode("utf-8", "ignore")
                    if tb.lstrip().startswith(("//", "/*", "///")):
                        err_loc["comment"] += 1
                    elif '"' in ctx[-60:] or tb.strip().startswith(('"', 'r"', 'b"')):
                        err_loc["string"] += 1
                    elif re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{1,}", tb):
                        err_loc["ident"] += 1
                    else:
                        err_loc["punct"] += 1
                try:
                    _d = bpe.vocab.get(b_, b"").decode("utf-8", "ignore")
                    if "\n" in _d:
                        ctx = _d[_d.rfind("\n") + 1:]
                    else:
                        ctx += _d
                except Exception:
                    pass
            tok_ok += sum(int(a == b) for a, b in zip(pred[k, :ln].tolist(), flat[k][:ln]))
            tot_tok += ln
            chunk_ok += int(pred[k, :ln].tolist() == flat[k][:ln])
            tot_b += 1
        for si, sn in enumerate(batch_sn):
            tot_s += 1
            if recs[si] == sn:
                snip_exact += 1
            elif len(worst) < 3 and len(recs[si]) > 40:
                worst.append((sn.replace("\n", " ")[:80], recs[si].replace("\n", " ")[:80]))
res = dict(snippets=tot_s, blocks=tot_b, token_acc=round(tok_ok / max(1, tot_tok), 4),
           err_by_location=err_loc,
           block_exact=round(chunk_ok / max(1, tot_b), 4),
           snippet_text_exact=round(snip_exact / max(1, tot_s), 4), secs=round(time.time() - t0))
print("HELDOUT", json.dumps(res), flush=True)
for a, b in worst:
    print("WORST:", a, "=>", b, flush=True)
json.dump(res, open(f"{LAB}/eval/rust-codec-heldout.json", "w"), indent=1)
