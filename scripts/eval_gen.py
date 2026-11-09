#!/usr/bin/env python
# B1 eval: sample unconditional stories from our trained generator (frozen official codec),
# score distinct-4, length, and GPT-2-large perplexity on CPU (community scorer class).
import json, os, sys, time
import torch
from omegaconf import OmegaConf

REPO = "/home/cobrajet/ncm-lab/ncm"; sys.path.insert(0, REPO)
from tokenization.vqvae_tokenizer import VQVAETokenizer
from models.next_concept import NextConceptTransformer

CKPT = os.environ.get("CKPT", "/home/cobrajet/ncm-lab/train/full1/gen_last.pt")
N = int(os.environ.get("N", "128"))
CK = "/home/cobrajet/ncm-lab/artifacts/tinystories"

CFG = OmegaConf.load(os.path.join(REPO, "config/eval/tinystories.yaml"))
CFG.tokenizer.vqvae.checkpoint_path = os.path.join(CK, CFG.checkpoints.tokenizer)
CFG.tokenizer.pre_tokenizer.vocab_file_path = os.path.join(CK, CFG.checkpoints.bpe_vocab)
CFG.tokenizer.pre_tokenizer.merges_file_path = os.path.join(CK, CFG.checkpoints.bpe_merges)
tk = VQVAETokenizer.from_config(CFG.tokenizer)
tk.vqvae = tk.vqvae.cuda().eval()

class NS: pass
mc = torch.load(CKPT, map_location="cpu", weights_only=False)
ns = NS()
for k, v in mc["cfg"].items(): setattr(ns, k, v)
net = NextConceptTransformer(vqvae_local=tk.vqvae, config=ns).cuda()
net.load_state_dict(mc["model"]); net.eval()
print("loaded ckpt steps_done:", mc.get("steps_done"), flush=True)

P = int(CFG.model.prefix_len); EOS = tk.pre_tokenizer.eos_token_id; PAD = tk.pre_tokenizer.pad_token_id
TEMP = float(CFG.generation.get("temperature", 1.0)); TOPK = int(CFG.generation.get("top_k", 50)); TOPP = float(CFG.generation.get("top_p", 0.9))
print("sampling:", TEMP, TOPK, TOPP, flush=True)

torch.manual_seed(1234)
stories = []
with torch.no_grad():
    for s in range(0, N, 32):
        B = min(32, N - s)
        pfx = torch.full((B, P), EOS, dtype=torch.long, device="cuda")
        l0 = torch.randint(0, 16, (B, 1), device="cuda")  # TS supplies level-0 codes; official eval uses ckpt unigram prior
        g = net.generate(l0, pfx, temperature=TEMP, top_k=TOPK, top_p=TOPP)
        _, zl = tk.vqvae.vq.idx_to_zhats(g, return_zlevels=True)
        ids = tk.vqvae.decoder(zl[-1]).argmax(-1)
        for row in ids.tolist():
            if EOS in row: row = row[:row.index(EOS)]
            stories.append(tk.pre_tokenizer.decode([t for t in row if t not in (EOS, PAD)], skip_special_tokens=True))
        print(f"sampled {len(stories)}", flush=True)

def distinct4(texts):
    tot, uniq = 0, set()
    for t in texts:
        w = t.lower().split()
        for i in range(len(w) - 3):
            tot += 1; uniq.add(tuple(w[i:i + 4]))
    return (len(uniq) / tot) if tot else 0.0

d4 = distinct4(stories)
words = [len(t.split()) for t in stories]

# GPT-2-large PPL on CPU (community reference scorer)
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
enc = AutoTokenizer.from_pretrained("gpt2-large")
lm = AutoModelForCausalLM.from_pretrained("gpt2-large", torch_dtype=torch.float32).eval()
enc.pad_token = enc.eos_token
nlls, toks = [], 0
t0 = time.time()
with torch.no_grad():
    for i in range(0, len(stories), 16):
        batch = [s for s in stories[i:i + 16] if s.strip()]
        enc_in = enc(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
        labels = enc_in.input_ids.masked_fill(enc_in.attention_mask == 0, -100)
        out = lm(**enc_in, labels=labels)
        n_tok = int((labels != -100).sum())
        nlls.append(float(out.loss) * n_tok); toks += n_tok
        print(f"ppl {len(nlls)}/{(len(stories)+15)//16} {time.time()-t0:.0f}s", flush=True)
ppl = float(np.exp(sum(nlls) / max(1, toks)))

res = {"n": len(stories), "distinct_4": d4, "mean_words": sum(words) / len(words),
       "min_words": min(words), "ppl_gpt2_large": ppl, "steps_done": mc.get("steps_done"),
       "params_M": sum(p.numel() for p in net.parameters()) / 1e6,
       "samples": stories[:8]}
os.makedirs(os.path.dirname("/home/cobrajet/ncm-lab/eval/b1-full1.json"), exist_ok=True)
json.dump(res, open("/home/cobrajet/ncm-lab/eval/b1-full1.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "samples"}, indent=1))
for t in stories[:6]:
    print("—", t[:220])
