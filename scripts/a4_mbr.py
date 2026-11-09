#!/usr/bin/env python
# A4 — best-of-N exchange rate (=L3): N in {1,4,20} sampled XSum candidates,
# judge (:8083 gpt-oss-20b, sees ARTICLE+CANDIDATES ONLY, no reference) as selector.
# Measures quality-per-extra-second of test-time compute: judge-pick vs random-pick
# vs oracle-pick vs N=1 mean, with gen+judge wall clocks.
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
N = int(os.environ.get("A_N", "128"))
CAND = int(os.environ.get("A_CAND", "20"))
NSUB = [n for n in (1, 4, 20) if n <= CAND]
TEMP, TOPK, TOPP = 1.0, 0, 1.0
OUTJ = os.environ.get("OUTJ", "/home/cobrajet/ncm-lab/eval/A4-mbr.json")
JUDGE_URL = os.environ.get("JUDGE_URL", "http://127.0.0.1:8083/v1/chat/completions")
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
print(f"loaded model+data {time.time()-t0:.0f}s  P={P} n={N}", flush=True)

def detok(ids):
    ids = [t for t in ids if t not in (EOS, PAD)]
    if EOS in list(ids):
        pass
    return tk.pre_tokenizer.decode(ids, skip_special_tokens=True)

def art_text(i):
    it = items[i]
    ids = list(it)[:1000] if not isinstance(it, str) else None
    return detok(ids) if ids is not None else it[:6000]

def cut_eos(ids):
    ids = list(ids)
    if EOS in ids:
        ids = ids[:ids.index(EOS)]
    return ids

cands = [[] for _ in range(N)]
gen_t = []
with torch.no_grad():
    for c in range(CAND):
        torch.manual_seed(1000 + c)
        tp = time.time()
        for s in range(0, N, 4):
            b = list(range(s, min(s + 4, N)))
            pfx = torch.tensor([ (list(items[i])[:P] + [PAD] * P)[:P] if not isinstance(items[i], str) else (items[i][:P] + [PAD] * P)[:P] for i in b], device="cuda")
            g = net.generate(None, pfx, temperature=TEMP, top_k=TOPK, top_p=TOPP, cfg_scale=0.0, cfg_null_eos=EOS)
            _, zl = tk.vqvae.vq.idx_to_zhats(g, return_zlevels=True)
            ids = tk.vqvae.decoder(zl[-1]).argmax(-1)
            for k, i in enumerate(b):
                cands[i].append(detok(cut_eos(ids[k].tolist())))
        gen_t.append(time.time() - tp)
        if c % 5 == 0 or c == CAND - 1:
            print(f"pass {c+1}/{CAND} {time.time()-t0:.0f}s", flush=True)
gen_secs_per_cand = sum(gen_t) / N
scores = [[rouge_l(refs[i] if isinstance(refs[i], str) else detok(cut_eos(list(refs[i]))), h) for h in cands[i]] for i in range(N)]
print(f"ROUGE scored {time.time()-t0:.0f}s", flush=True)

models = json.load(urllib.request.urlopen(JUDGE_URL.replace("/chat/completions", "/models")))["data"]
MODEL = models[0]["id"]

def judge(i, retry=False):
    arts = art_text(i)
    lst = "\n".join(f"{k+1}. {h}" for k, h in enumerate(cands[i]))
    body = {"model": MODEL, "temperature": 0.1, "max_tokens": 900, "reasoning_effort": "low",
            "messages": [{"role": "user", "content":
                "You will see a news article and 20 candidate one-sentence summaries. "
                "Rate each candidate 0-10 for how well it summarizes the article (accuracy + key facts covered; penalize invented facts).\n"
                f"ARTICLE:\n{arts[:4500]}\n\nCANDIDATES:\n{lst}\n\n"
                + ("The 20 numbers only, comma-separated. Nothing else." if retry else
                   "Respond with ONLY the 20 numbers separated by commas, in order.")}]}
    req = urllib.request.Request(JUDGE_URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        out = json.loads(urllib.request.urlopen(req, timeout=240).read())["choices"][0]["message"]["content"]
        nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", out)[:20]]
        if len(nums) == 20:
            return nums
        return judge(i, True) if not retry else None
    except Exception as e:
        print("judge fail", i, repr(e)[:100], flush=True)
        return judge(i, True) if not retry else None

jt = time.time()
with ThreadPoolExecutor(WORKERS) as ex:
    jres = list(ex.map(judge, range(N)))
judge_secs_total = time.time() - jt
judge_secs_per_prompt = judge_secs_total / N

rows = {}
import random
rng = random.Random(0)
for n in NSUB:
    jpicks, rpicks, opicks = [], [], []
    ok = 0
    for i in range(N):
        if jres[i] is not None:
            ok += 1
            j = max(range(n), key=lambda k: jres[i][k])
            jpicks.append(scores[i][j])
        rpicks.append(scores[i][rng.randrange(n)])
        opicks.append(max(scores[i][:n]))
    rows[f"N{n}"] = dict(
        mean_1=round(sum(scores[i][0] for i in range(N)) / N, 3),
        judge=round(sum(jpicks) / max(1, ok), 3), random=round(sum(rpicks) / N, 3),
        oracle=round(sum(opicks) / N, 3))
rows["judge_ok_prompts"] = sum(1 for r in jres if r is not None)
rows["gen_secs_per_prompt_per_cand"] = round(gen_secs_per_cand, 3)
rows["judge_secs_per_prompt"] = round(judge_secs_per_prompt, 2)
if "N20" in rows and "N20" in rows:
    rows["delta_judge_vs_random_N20"] = round(rows["N20"]["judge"] - rows["N20"]["random"], 3)
if "N20" in rows and "N1" in rows:
    rows["capture_N20"] = round((rows["N20"]["judge"] - rows["N1"]["mean_1"]) / max(1e-9, rows["N20"]["oracle"] - rows["N1"]["mean_1"]), 3)
json.dump(dict(summary=rows, sample_scores=scores[:8], judge_scores=[r[:8] if r else None for r in jres[:8]]), open(OUTJ, "w"), indent=1)
print("A4 " + json.dumps(rows), flush=True)
print("DONE", f"{time.time()-t0:.0f}s", flush=True)
