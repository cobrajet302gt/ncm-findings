#!/usr/bin/env python
# A4b — MBR selector RETEST with the confounds fixed: PAIRWISE bracket selection
# (groups of 5 -> finalist -> champion) + judge reasoning at default (medium), max_tokens 900.
# If capture is still ~0, the soft-judge MBR lane closes for good.
import json, os, pickle, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
import torch
from omegaconf import OmegaConf

REPO = "/home/cobrajet/ncm-lab/ncm"
sys.path.insert(0, REPO)
from tokenization.vqvae_tokenizer import VQVAETokenizer
from models.next_concept import NextConceptTransformer

CK = "/home/cobrajet/ncm-lab/artifacts/xsum"
DATA = "/home/cobrajet/ncm-lab/data/xsum"
N = int(os.environ.get("A_N", "32"))
CAND = 20
OUTJ = os.environ.get("OUTJ", "/home/cobrajet/ncm-lab/eval/A4b-mbr.json")
JUDGE_URL = "http://127.0.0.1:8083/v1/chat/completions"
WORKERS = 6
torch.manual_seed(0)

def norm(t):
    t = t.replace("<|endoftext|>", " ").replace("<|pad|>", " ")
    return " ".join(t.lower().replace("\n", " ").split())

def rouge_l(ref, hyp):
    a, b = norm(ref).split(), norm(hyp).split()
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b):
            cur[j + 1] = prev[j] + 1 if x == y else max(cur[j], prev[j + 1])
        prev = cur
    lcs = prev[len(b)]
    p, r = lcs / len(b), lcs / len(a)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)

t0 = time.time()
CFG = OmegaConf.load(os.path.join(REPO, "config/eval/xsum.yaml"))
CFG.tokenizer.vqvae.checkpoint_path = os.path.join(CK, CFG.checkpoints.tokenizer)
for key, name in (("vocab_file_path", "bpe_vocab"), ("merges_file_path", "bpe_merges")):
    if CFG.checkpoints.get(name):
        CFG.tokenizer.pre_tokenizer[key] = os.path.join(CK, CFG.checkpoints[name])
tk = VQVAETokenizer.from_config(CFG.tokenizer)
tk.vqvae = tk.vqvae.cuda().eval()
net = NextConceptTransformer(vqvae_local=tk.vqvae, config=CFG.model).cuda()
sd = torch.load(os.path.join(CK, CFG.checkpoints.generator), map_location="cpu", weights_only=False)
net.load_state_dict(sd["model"], strict=True)
net.eval()
P = int(CFG.model.prefix_len)
EOS, PAD = tk.pre_tokenizer.eos_token_id, tk.pre_tokenizer.pad_token_id
items = pickle.load(open(os.path.join(DATA, str(CFG.data["test"])), "rb"))[:N]
refs = pickle.load(open(os.path.join(DATA, str(CFG.data["test_refs"])), "rb"))[:N]

def detok(ids):
    return tk.pre_tokenizer.decode([t for t in ids if t not in (EOS, PAD)], skip_special_tokens=True)

def art_text(i):
    it = items[i]
    return detok(list(it)[:1000]) if not isinstance(it, str) else it[:6000]

def cut_eos(ids):
    ids = list(ids)
    if EOS in ids:
        ids = ids[:ids.index(EOS)]
    return ids

MODEL = json.load(urllib.request.urlopen(JUDGE_URL.replace("/chat/completions", "/models")))["data"][0]["id"]

def jcall(article, hs, question):
    lst = "\n".join(f"{k+1}. {h}" for k, h in enumerate(hs))
    body = {"model": MODEL, "temperature": 0.1, "max_tokens": 900,
            "messages": [{"role": "user", "content":
                "You will see a news article and candidate one-sentence summaries.\n"
                f"ARTICLE:\n{article[:4500]}\n\nCANDIDATES:\n{lst}\n\n" + question}]}
    req = urllib.request.Request(JUDGE_URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            out = json.loads(urllib.request.urlopen(req, timeout=240).read())["choices"][0]["message"]["content"]
            m = re.findall(r"\d+", out)
            if m and 1 <= int(m[-1]) <= len(hs):
                return int(m[-1]) - 1
        except Exception as e:
            print("judge fail", repr(e)[:90], flush=True)
        time.sleep(2)
    return 0

def gen():
    cands = [[] for _ in range(N)]
    with torch.no_grad():
        for c in range(CAND):
            torch.manual_seed(1000 + c)
            for s in range(0, N, 4):
                b = list(range(s, min(s + 4, N)))
                pfx = torch.tensor([(list(items[i])[:P] + [PAD] * P)[:P] if not isinstance(items[i], str) else (items[i][:P] + [PAD] * P)[:P] for i in b], device="cuda")
                g = net.generate(None, pfx, temperature=1.0, top_k=0, top_p=1.0, cfg_scale=0.0, cfg_null_eos=EOS)
                _, zl = tk.vqvae.vq.idx_to_zhats(g, return_zlevels=True)
                ids = tk.vqvae.decoder(zl[-1]).argmax(-1)
                for k, i in enumerate(b):
                    cands[i].append(detok(cut_eos(ids[k].tolist())))
            if c % 5 == 0:
                print(f"gen pass {c+1}/{CAND} {time.time()-t0:.0f}s", flush=True)
    return cands

import random
cache = "/home/cobrajet/ncm-lab/eval/A4b-cands.json"
if os.path.exists(cache):
    cands = json.load(open(cache))
    print("candidates from cache", flush=True)
else:
    cands = gen()
    json.dump(cands, open(cache, "w"))
scores = [[rouge_l(refs[i] if isinstance(refs[i], str) else detok(cut_eos(list(refs[i]))), h) for h in cands[i]] for i in range(N)]

def bracket(i):
    art = art_text(i)
    pool = list(range(CAND))
    rnd = 0
    while len(pool) > 1:
        rnd += 1
        pairs = [(pool[j], pool[j + 1]) for j in range(0, len(pool) - 1, 2)]
        if len(pool) % 2:
            winners = [p[0] for p in pairs] + [pool[-1]]
            # pair matches in parallel
            with ThreadPoolExecutor(min(WORKERS, len(pairs))) as ex:
                outs = list(ex.map(lambda pr: jcall(art, [cands[i][pr[0]], cands[i][pr[1]]],
                       "Two summaries. Reply with ONLY the number (1 or 2) of the BETTER summary, on the last line as 'Choice: N'."), pairs))
            winners = [pr[o] for pr, o in zip(pairs, outs)] + [pool[-1]]
        else:
            with ThreadPoolExecutor(min(WORKERS, len(pairs))) as ex:
                outs = list(ex.map(lambda pr: jcall(art, [cands[i][pr[0]], cands[i][pr[1]]],
                       "Two summaries. Reply with ONLY the number (1 or 2) of the BETTER summary, on the last line as 'Choice: N'."), pairs))
            winners = [pr[o] for pr, o in zip(pairs, outs)]
        pool = winners
    return pool[0]

jt = time.time()
with ThreadPoolExecutor(4) as ex:  # prompts x internal pair parallelism; keep judge sane
    champ = list(ex.map(bracket, range(N)))
judge_secs = time.time() - jt
rng = random.Random(0)
mean1 = sum(scores[i][0] for i in range(N)) / N
jc = sum(scores[i][champ[i]] for i in range(N)) / N
rc = sum(scores[i][rng.randrange(CAND)] for i in range(N)) / N
oc = sum(max(scores[i]) for i in range(N)) / N
rows = dict(N1_mean=round(mean1, 3), judge_champ=round(jc, 3), random_champ=round(rc, 3),
            oracle=round(oc, 3), capture=round((jc - mean1) / max(1e-9, oc - mean1), 3),
            judge_secs_total=round(judge_secs, 1), judge_secs_per_prompt=round(judge_secs / N, 1))
json.dump(dict(summary=rows, champ=champ, scores=scores, cands=cands), open(OUTJ, "w"))
print("A4b " + json.dumps(rows), flush=True)
print("DONE", f"{time.time()-t0:.0f}s", flush=True)
