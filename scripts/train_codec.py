#!/usr/bin/env python
# Stage-1 Rust codec trainer. Upstream implements every piece EXCEPT the driver loop:
# BPETokenizer (byte-tuple vocab/merges pickles), VQVAE (multiscale residual VQ, EMA
# codebooks, kmeans_init_level in quant.py). This driver: fit BPE-8192 on Rust -> segment
# stream -> kmeans-init -> CE+vq_loss training -> artifacts/rustcodec/ laid out like
# artifacts/tinystories/ so stage-2 (train_gen.py) + audits load it unchanged.
# Env: STEPS=20000 BS=128 LR=4e-4 WARM=500 RESTART=0.0 VOCAB=8192 OUT CORPUS
import glob, json, os, pickle, random, re, sys, time
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

REPO = "/home/cobrajet/ncm-lab/ncm"
sys.path.insert(0, REPO)
from models.vqvae import VQVAE
from tokenization.bpe_tokenizer import BPETokenizer

LAB = "/home/cobrajet/ncm-lab"
OUT = os.environ.get("OUT", f"{LAB}/artifacts/rustcodec")
CORPUS = os.environ.get("CORPUS", f"{LAB}/data/rust/corpus")
STEPS = int(os.environ.get("STEPS", "20000"))
BS = int(os.environ.get("BS", "128"))
AMUL = int(os.environ.get("AMUL", "1"))  # micro-batches: effective batch = BS*AMUL at BS-sized graph
LR = float(os.environ.get("LR", "4e-4"))
RESTART = float(os.environ.get("RESTART", "0.0"))
VOCAB = int(os.environ.get("VOCAB", "8192"))
SEG = 128
torch.manual_seed(0)
random.seed(0)
os.makedirs(OUT, exist_ok=True)
t0 = time.time()

# ---------- corpus ----------
files = sorted(glob.glob(f"{CORPUS}/**/*.rs", recursive=True))
if not files:
    print("corpus empty — fetch_rust_corpus still running; abort"); sys.exit(1)
texts = []
for f in files:
    try:
        t = open(f, encoding="utf-8", errors="ignore").read()
        if len(t) > 100:
            texts.append(t)
    except Exception:
        pass
print(f"corpus {len(files)} files, {sum(map(len, texts))/1e6:.1f} MB", flush=True)

# ---------- BPE fit (tokenizers lib) -> repo byte-tuple format ----------
VOCF, MRGF = f"{OUT}/bpe_vocab_8192.pickle", f"{OUT}/bpe_merges_8192.pickle"
if not os.path.exists(VOCF):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Split(
        r"(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+", behavior="contiguous")
    tr = trainers.BpeTrainer(vocab_size=VOCAB, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                             show_progress=False, min_frequency=2)
    tok.train_from_iterator(texts[: max(3, len(texts) // 2)], tr)
    # invert GPT-2 byte-level unicode table: token string -> bytes
    vis = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    b2u = {b: chr(b) if b in vis else chr(b + 256) for b in range(256)}
    u2b = {v: k for k, v in b2u.items()}
    def s2bytes(s):
        return bytes(u2b[c] for c in s)
    vocab = tok.get_vocab()
    voc = {}
    for s, i in sorted(vocab.items(), key=lambda kv: kv[1]):
        try:
            bkey = s2bytes(s)
            if bkey not in {v for v in voc.values()}:
                voc[len(voc)] = bkey
        except KeyError:
            pass
    for sp in ("<|endoftext|>", "<|pad|>"):
        if sp.encode() not in set(voc.values()):
            voc[len(voc)] = sp.encode()
    have = set(voc.values())
    for b in range(256):  # every byte must be a token or encode() KeyErrors on rare utf-8
        bb = bytes([b])
        if bb not in have:
            voc[len(voc)] = bb
            have.add(bb)
    merges = []
    for pair in json.loads(tok.to_str())["model"].get("merges", []):
        a, b = pair if isinstance(pair, (list, tuple)) else pair.split(" ")
        try:
            merges.append((s2bytes(a), s2bytes(b)))
        except KeyError:
            pass
    pickle.dump(voc, open(VOCF, "wb"))
    pickle.dump(merges, open(MRGF, "wb"))
    print(f"BPE fit {time.time()-t0:.0f}s vocab={len(voc)} merges={len(merges)}", flush=True)

class Cfg:
    vocab_file_path, merges_file_path = VOCF, MRGF
    special_tokens = {"eos_token": "<|endoftext|>", "pad_token": "<|pad|>"}
bpe = BPETokenizer.from_config(Cfg())
_rv = {k: v for k, v in bpe.reverse_vocab.items()} if all(isinstance(k, bytes) for k in bpe.reverse_vocab) else bpe.reverse_vocab
EOS, PAD = _rv[b"<|endoftext|>"], _rv[b"<|pad|>"]
sample_ids = bpe.encode(texts[0][:500])
print(f"bpe via repo class: vocab={bpe.vocab_size} eos={EOS} pad={PAD} sample_len={len(sample_ids)}", flush=True)
VOCAB = bpe.vocab_size  # actual vocab, not the 8192 ceiling
rt = bpe.decode(sample_ids)
print(f"roundtrip head ok={rt[:80] == texts[0][:80]}", flush=True)

# ---------- segment stream ----------
CACHE = f"{OUT}/segments.pt"
if not os.path.exists(CACHE):
    segs = []
    for t in texts:
        ids = bpe.encode(t)
        for s in range(0, max(1, len(ids) - SEG + 1), SEG):
            w = list(ids[s:s + SEG])
            if len(w) == SEG:
                segs.append(w)
    segs = torch.tensor(segs, dtype=torch.long)
    print(f"segments {segs.shape[0]} x {SEG} ({time.time()-t0:.0f}s)", flush=True)
    torch.save(segs, CACHE)
segs = torch.load(CACHE)
dev = "cuda"

# ---------- VQVAE (mirror tinystories config exactly; same arch proven on prose) ----------
base = OmegaConf.load(f"{LAB}/artifacts/tinystories/config.yaml")
tvq = OmegaConf.to_container(base.tokenizer.vqvae, resolve=True)
vqcfg = base.tokenizer.vqvae.vector_quantizer_config  # keep DictConfig (ctor does .attr access)
vqvae = VQVAE(in_vocab_size=VOCAB, context_length=SEG, embed_dim=tvq["embed_dim"],
              vqvae_vocab_size=tvq["vqvae_vocab_size"], n_layers=tvq["n_layers"],
              vector_quantizer_config=vqcfg, compression_factor=tvq["compression_factor"],
              dropout=tvq["dropout"], pre_quant_groupnorm=tvq["pre_quant_groupnorm"],
              use_qk_norm=tvq["use_qk_norm"], pre_quant_dropout=tvq["pre_quant_dropout"]).to(dev)
if RESTART > 0:
    vqvae.vq.restart_threshold = RESTART  # issue #16 fix must be patched in quant.py before enabling
opt = torch.optim.AdamW([p for p in vqvae.parameters() if p.requires_grad], lr=LR)

def batch():
    ix = torch.randint(0, segs.shape[0], (BS,))
    return segs[ix].to(dev)

# kmeans-init all levels from live latents
vqvae.eval()
with torch.no_grad():
    z = vqvae.encode_z(batch())
KLVL = len(vqcfg["levels"]["manual_levels"])
for lvl in range(KLVL):
    vqvae.vq.kmeans_init_level(lvl, z.reshape(-1, vqvae.embed_dim)[:8192].cpu())
print(f"kmeans init {KLVL} levels done {time.time()-t0:.0f}s", flush=True)

# ---------- train ----------
vqvae.train()
opt.zero_grad(set_to_none=True)
curve = []
for step in range(1, STEPS + 1):
    for _mu in range(AMUL):
        x = batch()
        logits, vq_loss, util_pl, util_g, _ = vqvae(x)
        ce = F.cross_entropy(logits[:, 1:].reshape(-1, VOCAB), x[:, 1:].reshape(-1), ignore_index=PAD)
        ((ce + vq_loss) / AMUL).backward()
    torch.nn.utils.clip_grad_norm_(vqvae.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    if step % 250 == 0 or step == STEPS:
        vqvae.eval()
        with torch.no_grad():
            xu = batch()
            lg, vl, upl, ug, _ = vqvae(xu)
            ceu = F.cross_entropy(lg[:, 1:].reshape(-1, VOCAB), xu[:, 1:].reshape(-1), ignore_index=PAD)
            tok_acc = float((lg.argmax(-1)[:, 1:] == xu[:, 1:]).float().mean())
            chunk_exact = float((lg.argmax(-1)[:, 1:] == xu[:, 1:]).all(dim=1).float().mean())
            util = float(torch.tensor(upl).float().mean())
        vqvae.train()
        curve.append(dict(step=step, ce=round(float(ceu), 3), vq=round(float(vl), 3),
                          tok_acc=round(tok_acc, 4), chunk_exact=round(chunk_exact, 4), util=round(util, 3)))
        print(f"step {step} ce={float(ceu):.3f} tok_acc={tok_acc:.3f} chunk_exact={chunk_exact:.3f} util={util:.3f} ({time.time()-t0:.0f}s)", flush=True)
        torch.cuda.empty_cache()
        json.dump(curve, open(f"{OUT}/train_curve.json", "w"), indent=1)
        torch.save({"model": vqvae.state_dict()}, f"{OUT}/tokenizer_last.pt")

# ---------- save in artifacts layout ----------
torch.save({"model": vqvae.state_dict()}, f"{OUT}/tokenizer.pt")
cfg_out = OmegaConf.to_container(base, resolve=True)
cfg_out["tokenizer"]["vqvae"].pop("checkpoint_path", None)
cfg_out["tokenizer"]["vqvae"]["in_vocab_size"] = VOCAB
cfg_out["tokenizer"]["pre_tokenizer"]["vocab_file_path"] = os.path.basename(VOCF)
cfg_out["tokenizer"]["pre_tokenizer"]["merges_file_path"] = os.path.basename(MRGF)
cfg_out["checkpoints"] = {"generator": None, "tokenizer": "tokenizer.pt",
                          "bpe_vocab": os.path.basename(VOCF), "bpe_merges": os.path.basename(MRGF)}
OmegaConf.save(OmegaConf.create(cfg_out), f"{OUT}/config.yaml")
print("DONE", f"{time.time()-t0:.0f}s", flush=True)
