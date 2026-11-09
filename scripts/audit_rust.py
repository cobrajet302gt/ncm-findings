#!/usr/bin/env python
# Rust codec audit — THE gate for the Rust-NCM lane.
# M1: token inflation: Rust through Rust-fit BPE-8192 vs through TinyStories prose BPE
# M2: idiom fragmentation: pieces per tricky Rust construct (lifetimes, turbofish, macros)
# M3: frozen PROSE codec round-trip of Rust snippets (the real codec-tax number, domain-mismatched)
import glob, json, os, random, re, sys, time
import torch

LAB = "/home/cobrajet/ncm-lab"
sys.path.insert(0, f"{LAB}/ncm")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

# ---------- corpus ----------
files = [f for f in glob.glob("/home/cobrajet/strata/**/*.rs", recursive=True) if "target" not in f]
files += glob.glob(f"{LAB}/data/rust/crates/**/*.rs", recursive=True)
texts = []
for f in files:
    try:
        t = open(f, encoding="utf-8", errors="ignore").read()
        if 200 < len(t) < 200_000:
            texts.append(t)
    except Exception:
        pass
random.Random(0).shuffle(texts)
train = "\n\n".join(texts[: int(len(texts) * 0.8)])
audit = [t for t in texts[int(len(texts) * 0.8):]]
# snippets: cut functions-ish blocks for round-trip
snips = []
for t in audit:
    for m in re.finditer(r"(?:pub |)(?:async fn|fn|macro_rules!|impl|struct|enum|trait)\s[\s\S]{0,800}?\n\}", t):
        s = m.group(0)
        if 120 < len(s) < 900:
            snips.append(s)
random.Random(1).shuffle(snips)
snips = snips[:120]
print(f"corpus: {len(files)} files, {len(train)//1024} KiB train, {len(snips)} audit snippets", flush=True)

# ---------- M1/M2: BPE comparison ----------
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
t0 = time.time()
rust_bpe = Tokenizer(models.BPE())
rust_bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tr = trainers.BpeTrainer(vocab_size=8192, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                         show_progress=False)
rust_bpe.train_from_iterator(train.splitlines(), tr)
print(f"rust BPE fit {time.time()-t0:.0f}s", flush=True)

from omegaconf import OmegaConf
from tokenization.vqvae_tokenizer import VQVAETokenizer
CFG = OmegaConf.load(f"{LAB}/ncm/config/eval/xsum.yaml")
CK = f"{LAB}/artifacts/xsum"
CFG.tokenizer.vqvae.checkpoint_path = os.path.join(CK, CFG.checkpoints.tokenizer)
for key, name in (("vocab_file_path", "bpe_vocab"), ("merges_file_path", "bpe_merges")):
    if CFG.checkpoints.get(name):
        CFG.tokenizer.pre_tokenizer[key] = os.path.join(CK, CFG.checkpoints[name])
tk = VQVAETokenizer.from_config(CFG.tokenizer)
prose_bpe = tk.pre_tokenizer  # HF fast tokenizer

sample = "\n".join(snips)
n_rust = len(rust_bpe.encode(sample).ids)
n_prose = len(list(prose_bpe.encode(sample)))
n_char = len(sample)
IDIONS = ["'static", "'a ", "::<", "macro_rules!", "#[derive", "impl Iterator", "where",
          "unsafe fn", "async move", "dyn ", "-> Box<", "<T: ", "Self::", "loop {", "match "]
frag = {}
for idm in IDIONS:
    cnt = [0, 0]
    for t in audit:
        for _ in range(t.count(idm)):
            ids = rust_bpe.encode(idm).ids
            cnt[0] += 1
            cnt[1] += len(ids)
            break
    frag[idm] = round(cnt[1] / max(1, cnt[0]), 2)
m1 = dict(chars=n_char, vocab=rust_bpe.get_vocab_size(), tokens_rust_bpe=n_rust, tokens_prose_bpe=n_prose,
          chars_per_tok_rust=round(n_char / n_rust, 2), chars_per_tok_prose=round(n_char / n_prose, 2),
          inflation=round(n_prose / n_rust, 2))
print("M1/M2", json.dumps(dict(m1=m1, frag=frag)), flush=True)

# ---------- M3: frozen prose codec round-trip (per-level exactness depth included) ----------
t1 = time.time()
tk.vqvae = tk.vqvae.eval()  # CPU: share GPU1 with A4b

def _codes(text):
    idx = torch.tensor(tk._adjust_length(tk.pre_tokenizer.encode(text)), dtype=torch.long)
    return tk.encode_pre_tokenized_idx(idx)

def _recon_ms(text):
    return tk.decode_multiscales(_codes(text))

tk.decode_multiscales = _recon_ms if not hasattr(tk, "decode_multiscales") else tk.decode_multiscales
full_exact = 0
level_first = []
worst = []
with torch.no_grad():
    for k, snip in enumerate(snips):
        codes = _codes(snip)
        final = tk.decode(codes[0].tolist()).strip()
        ok = int(final == snip.strip())
        full_exact += ok
        if not ok and len(worst) < 3:
            worst.append((snip.replace("\n", " ")[:90], final.replace("\n", " ")[:90]))
        try:
            recs = tk.decode_multiscales(codes)
            if recs and isinstance(recs[0], (list, tuple)):
                recs = recs[0]
            lf = next((j for j, rtxt in enumerate(recs) if str(rtxt).strip() == snip.strip()), None)
        except Exception:
            lf = None
        level_first.append(lf)
        if (k + 1) % 40 == 0:
            print(f"M3 {k+1}/{len(snips)} {time.time()-t1:.0f}s", flush=True)
K = len(level_first)
n_exact = sum(1 for x in level_first if x is not None)
import collections
lvl_hist = dict(sorted(collections.Counter(x for x in level_first if x is not None).items()))
m3 = dict(snippets=len(snips), full_exact=round(full_exact / len(snips), 3),
          any_level_exact=round(n_exact / len(snips), 3), exact_at_level_hist=lvl_hist,
          median_exact_level=(sorted([x for x in level_first if x is not None])[n_exact // 2] if n_exact else None),
          secs=round(time.time() - t1))
print("M3", json.dumps(m3), flush=True)
for a, b in worst:
    print("WORST:", a, "=>", b, flush=True)
json.dump(dict(m1=m1, frag=frag, m3=m3, corpus=dict(files=len(files), train_kib=len(train) // 1024,
          snippets=len(snips))), open(f"{LAB}/eval/rust-codec-audit.json", "w"), indent=1)
print("DONE", flush=True)
