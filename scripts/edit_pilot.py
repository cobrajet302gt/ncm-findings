#!/usr/bin/env python
# B6 EDIT PILOT — first model-editing experiment on an NCM (as far as public work goes).
# Fact-swaps ("Tim" -> "Rex") taught to our 12.4M TinyStories generator (full1) four ways:
#   ctrl: no edit | roml: rank-one ROME-style update on blocks[l].mlp.c_proj
#   lora: LoRA r=8 on c_proj of middle layers | ft: small full-weight fine-tune
# Scored the KE way: efficacy (CE flip + top-1 on edited facts), generalization (other
# stories with same subject), specificity (control CE drift + sampled-text side effects),
# plus a codec-fidelity audit (Red-fuchs lesson: edits must survive the frozen codec).
# Forward/conditioning bookkeeping cloned verbatim from train_gen.py.
import json, os, pickle, re, sys, time, copy
import torch, torch.nn.functional as F
from omegaconf import OmegaConf

REPO = "/home/cobrajet/ncm-lab/ncm"; sys.path.insert(0, REPO)
from tokenization.vqvae_tokenizer import VQVAETokenizer
from models.next_concept import NextConceptTransformer

CK = "/home/cobrajet/ncm-lab/artifacts/tinystories"
NFACTS = int(os.environ.get("NFACTS", "200"))
NCTRL = int(os.environ.get("NCTRL", "100"))
NGEN = int(os.environ.get("NGEN", "20"))
NNEIGH = int(os.environ.get("NNEIGH", "160"))
NSAMP = int(os.environ.get("NSAMP", "32"))
LORA_STEPS = int(os.environ.get("LORA_STEPS", "300"))
FT_STEPS = int(os.environ.get("FT_STEPS", "150"))
TRACE_STEPS = int(os.environ.get("TRACE_STEPS", "30"))
TR_NFACTS = int(os.environ.get("TR_NFACTS", "8"))
LAY_CAND = [int(x) for x in os.environ.get("LAY_CAND", "1,2,3,4").split(",")]
LORA_R = int(os.environ.get("LORA_R", "8"))
LORA_SCALE = float(os.environ.get("LORA_SCALE", "0.1"))
LORA_LAYERS = [int(x) for x in os.environ.get("LORA_LAYERS", os.environ.get("LAY_CAND", "1,2,3,4")).split(",")]
LORA_LR = float(os.environ.get("LORA_LR", "1e-3"))
ARMS = [a.strip() for a in os.environ.get("ARMS", "roml,lora,ft").split(",") if a.strip()]
GENVIEWS = int(os.environ.get("GENVIEWS", "0"))
LOC = float(os.environ.get("LOC", "0.0"))
SEED = int(os.environ.get("SEED", "0"))
OUTJ = os.environ.get("OUTJ", "/home/cobrajet/ncm-lab/eval/edit-pilot.json")
torch.manual_seed(SEED)

CFG = OmegaConf.load(os.path.join(REPO, "config/eval/tinystories.yaml"))
CFG.tokenizer.vqvae.checkpoint_path = os.path.join(CK, CFG.checkpoints.tokenizer)
CFG.tokenizer.pre_tokenizer.vocab_file_path = os.path.join(CK, CFG.checkpoints.bpe_vocab)
CFG.tokenizer.pre_tokenizer.merges_file_path = os.path.join(CK, CFG.checkpoints.bpe_merges)
tk = VQVAETokenizer.from_config(CFG.tokenizer)
tk.vqvae = tk.vqvae.cuda().eval()
for p in tk.vqvae.parameters():
    p.requires_grad_(False)
CKPT = torch.load("/home/cobrajet/ncm-lab/train/full1/gen_last.pt", map_location="cpu", weights_only=False)
class NS: pass
ns = NS()
for k, v in CKPT["cfg"].items(): setattr(ns, k, v)
net0 = NextConceptTransformer(vqvae_local=tk.vqvae, config=ns).cuda().eval()
net0.load_state_dict(CKPT["model"])
for p in net0.parameters(): p.requires_grad_(False)

CTX = int(CFG.tokenizer.vqvae.context_length)
vq = tk.vqvae.vq
LEVELS = list(vq.levels); K = len(LEVELS)
OFF = []; _o = 0
for Lx in LEVELS: OFF.append(_o); _o += Lx
L = _o
EOS = tk.pre_tokenizer.eos_token_id; PAD = tk.pre_tokenizer.pad_token_id
P = int(CFG.model.prefix_len)
LVL_FIRST = getattr(net0, "_lvl_first", 1)
QUERY_L0 = bool(getattr(net0, "query_level0", True))
dev = torch.device("cuda")
NLAYER = int(ns.n_layer)
print(f"K={K} L={L} CTX={CTX} layers={NLAYER} QUERY_L0={QUERY_L0} P={P} LVL_FIRST={LVL_FIRST}", flush=True)

rows = pickle.load(open("/home/cobrajet/ncm-lab/data/tinystories/train_bpe_8192.pickle", "rb"))
rows = [r for r in rows if 8 <= len(r) <= 120]
texts = [tk.pre_tokenizer.decode(r) for r in rows]
NAME = re.compile(r"\b[A-Z][a-z]{2,9}\b")
POOL = ["Rex", "Bella", "Coco", "Lola", "Buddy", "Daisy", "Milo", "Nala", "Finn", "Ziggy",
        "Pixel", "Mochi", "Juno", "Kiko", "Taco", "Pipin", "Suki", "Boji", "Yuki", "Dex"]

def enc_ids(ids):
    adj = tk._adjust_length(list(ids))
    seg = adj[0] if isinstance(adj[0], list) else adj
    if isinstance(seg, tuple): seg = seg[0]
    if len(seg) != CTX: return None
    return tk.encode_pre_tokenized_idx(torch.tensor([seg], dtype=torch.long, device=dev))[0]

def enc_text(t):
    ids = tk.pre_tokenizer.encode(t)
    if ids and isinstance(ids[0], (list, tuple)): ids = list(ids[0])
    return enc_ids(ids)

def flat_at(codes, k):
    codes = codes.reshape(1, -1)
    return codes[:, OFF[k]:OFF[k] + LEVELS[k]]

def level_forward(net, codes, j, site_layer=None, site_patch=None, want_grad=True):
    """Teacher-forced logits for level j (conditions on j's own clean codes up to level j-1).
    Returns logits (1, LEVELS[j], VC). site_patch: tensor (1,1,384) replacing c_proj output at position p."""
    pfx = net.transformer.wte(tk.vqvae.encoder.wte(torch.full((1, P), EOS, dtype=torch.long, device=dev))[:, :P, :]).detach()
    la_init = LEVELS[0] if QUERY_L0 else 0
    z_hat = torch.zeros(1, LEVELS[-1], tk.vqvae.embed_dim, device=dev)
    lv = j - 1
    x = None
    hook = None
    if site_patch is not None:
        p = site_layer[1]
        lay = net.transformer.blocks[site_layer[0]].mlp.c_proj
        def hk(m, inp, out, p=p, v=site_patch):
            o = out.clone(); o[:, p, :] = v; return o
        hook = lay.register_forward_hook(hk)
    try:
        for k in range(0, lv + 1):
            c_k = flat_at(codes, k)
            grad_this = (k == lv and want_grad)
            ctx = torch.enable_grad() if grad_this else torch.no_grad()
            with ctx:
                z_q = vq.embeddings[k](c_k)
                z_hat, x = vq.get_next_autoregressive_input(k, z_hat, z_q)
        if lv < 0:
            raise ValueError("level 0 not predicted by this model")
        if net.use_post_upsample_conv:
            x = net._apply_post_upsample_conv_single(x, lv + (1 - LVL_FIRST))
        la = la_init + sum(LEVELS[1:lv + 1])
        if net.config.use_level_encoding:
            lpk = net.transformer.wle(net.lvl_1L[:, la:la + LEVELS[j]])
            x = net.transformer.wte(x) + lpk[:, :x.shape[1], :]
        else:
            x = net.transformer.wte(x)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = net._decode_level_logits(x, pfx, lv, None, hidx=lv + (1 - LVL_FIRST))
    finally:
        if hook is not None: hook.remove()
    return logits[0].float()

def ce_at(net, codes, mask, j, site=None, patch=None, tgt_codes=None):
    logits = level_forward(net, codes, j, site_layer=(site + (patch,) if patch is not None else None) if site is not None else None,
                           site_patch=patch, want_grad=(patch is not None) or torch.is_grad_enabled())
    tgt = flat_at(tgt_codes if tgt_codes is not None else codes, j)[0]
    m = mask[OFF[j]:OFF[j] + LEVELS[j]]
    if int(m.sum()) == 0: return None, logits
    return F.cross_entropy(logits[m], tgt[m]), logits

def level_keys(net, codes, j, layer):
    """c_proj inputs (keys) at level j, graph of layer `layer`."""
    cap = {}
    lay = net.transformer.blocks[layer].mlp.c_proj
    h = lay.register_forward_pre_hook(lambda m, inp: cap.setdefault("k", inp[0].detach()))
    with torch.no_grad():
        level_forward(net, codes, j, want_grad=False)
    h.remove()
    return cap["k"][0]  # (LEVELS[j], 1536)

# ---------- facts ----------
t0 = time.time()
name_index = {}
for i, t in enumerate(texts):
    for m in set(NAME.findall(t)):
        name_index.setdefault(m, []).append(i)
def enc_row(i): return enc_ids(rows[i])
facts = []
for i in range(len(texts)):
    if len(facts) >= NFACTS * 2: break
    t = texts[i]; A = NAME.findall(t)
    if len(A) != 1 or t.count(A[0]) != 1: continue
    A = A[0]
    if A in POOL: continue
    if len(name_index.get(A, [])) < 2: continue
    B = next((b for b in POOL if b not in t), None)
    if B is None: continue
    ca = enc_row(i)
    cb = enc_text(re.sub(r"\b" + re.escape(A) + r"\b", B, t, count=1))
    if ca is None or cb is None: continue
    d = (ca != cb); d[:OFF[1]] = False
    if int(d[OFF[1]:].sum()) == 0: continue
    facts.append(dict(i=i, A=A, B=B, codes_A=ca, codes_B=cb, D=d))
def dec_top(codes):
    with torch.no_grad():
        _, zl = vq.idx_to_zhats(codes[None])
        ids = tk.vqvae.decoder(zl[-1]).argmax(-1)[0].tolist()
    return tk.pre_tokenizer.decode([x for x in ids if x not in (EOS, PAD)], skip_special_tokens=True)
audit_ok, keep = 0, []
n_cand = len(facts)
for f in facts:
    if f["B"] in dec_top(f["codes_B"]): audit_ok += 1; keep.append(f)
facts = keep[:NFACTS]
def best_level(d):
    return max(range(1, K), key=lambda j: int(d[OFF[j]:OFF[j] + LEVELS[j]].sum()))
for f in facts: f["Dlvl"] = best_level(f["D"])
print(f"candidates={n_cand} kept_after_audit={audit_ok} ({time.time()-t0:.0f}s)", flush=True)

ABset = set()
for f in facts: ABset.add(f["A"]); ABset.add(f["B"])
CTRL = []
for i, t in enumerate(texts):
    if len(CTRL) >= NCTRL: break
    if any(x in ABset for x in NAME.findall(t)): continue
    cc = enc_row(i)
    if cc is None: continue
    CTRL.append(dict(i=i, codes=cc))
NEIGH = []
for i in torch.randint(0, len(rows), (NNEIGH * 2,), generator=torch.Generator().manual_seed(7)).tolist():
    if len(NEIGH) >= NNEIGH: break
    cc = enc_row(int(i))
    if cc is not None: NEIGH.append(dict(codes=cc))
genpairs = []
for f in facts:
    if len(genpairs) >= NGEN: break
    j2 = [r for r in name_index[f["A"]] if r != f["i"]]
    if not j2: continue
    i2 = j2[0]
    ca2 = enc_row(i2)
    cb2 = enc_text(re.sub(r"\b" + re.escape(f["A"]) + r"\b", f["B"], texts[i2], count=1))
    if ca2 is None or cb2 is None: continue
    d2 = (ca2 != cb2); d2[:OFF[1]] = False
    if int(d2[OFF[1]:].sum()) == 0: continue
    genpairs.append(dict(A=f["A"], B=f["B"], codes_A=ca2, codes_B=cb2, D=d2, Dlvl=best_level(d2), held=True))
# TRAIN-time generalization views: 2 extra stories per fact, included in LoRA batches when GENVIEWS>0
gviews = []
for f in facts:
    extra = [r for r in name_index[f["A"]] if r != f["i"]][2:4]
    for i2 in extra:
        ca2 = enc_row(i2)
        cb2 = enc_text(re.sub(r"\b" + re.escape(f["A"]) + r"\b", f["B"], texts[i2], count=1))
        if ca2 is None or cb2 is None: continue
        d2 = (ca2 != cb2); d2[:OFF[1]] = False
        if int(d2[OFF[1]:].sum()) == 0: continue
        gviews.append(dict(A=f["A"], B=f["B"], codes_A=ca2, codes_B=cb2, D=d2, Dlvl=best_level(d2), held=False))
print(f"train views={len(gviews)} heldout gen={len(genpairs)} ({time.time()-t0:.0f}s)", flush=True)
print(f"ctrl={len(CTRL)} neigh={len(NEIGH)} gen={len(genpairs)} ({time.time()-t0:.0f}s)", flush=True)

# ---------- scoring ----------
def fact_scores(net, fl, cap=60):
    effs, hits, tot = [], 0, 0
    cleans = []
    with torch.no_grad():
        for f in fl[:cap]:
            j = f["Dlvl"]; m = f["D"][OFF[j]:OFF[j] + LEVELS[j]]
            cA, lgA = ce_at(net, f["codes_A"], f["D"], j)
            cB, lgB = ce_at(net, f["codes_B"], f["D"], j)
            effs.append(float(cA) - float(cB))
            # length-clean: SAME conditioning (A's sequence), A-targets vs B-targets
            cAA, _ = ce_at(net, f["codes_A"], f["D"], j, tgt_codes=f["codes_A"])
            cBA, _ = ce_at(net, f["codes_A"], f["D"], j, tgt_codes=f["codes_B"])
            cleans.append(float(cAA) - float(cBA))
            tB = flat_at(f["codes_B"], j)[0]
            hits += int((lgB.argmax(-1)[m] == tB[m]).sum()); tot += int(m.sum())
    return dict(pref_delta=sum(effs) / max(1, len(effs)), top1_B=hits / max(1, tot),
                clean_pref=sum(cleans) / max(1, len(cleans)))

def ctrl_ce_all(net, c):
    ce, n = 0.0, 0
    with torch.no_grad():
        for j in range(1, K):
            r = ce_at(net, c["codes"], torch.ones(L, dtype=torch.bool, device=dev), j)[0]
            if r is not None: ce += float(r); n += 1
    return ce / max(1, n)
def ctrl_drift(net, base):
    return sum(ctrl_ce_all(net, c) - base[c["i"]] for c in CTRL) / max(1, len(CTRL))
def sample_texts(net, n=NSAMP, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b in range(0, n, 8):
            B = min(8, n - b)
            l0 = torch.randint(0, 16, (B, 1), device="cuda", generator=g)
            pfx = torch.full((B, P), EOS, dtype=torch.long, device=dev)
            fl = net.generate(l0, pfx, temperature=1.0, top_k=0)
            _, zl = vq.idx_to_zhats(fl)
            ids = tk.vqvae.decoder(zl[-1]).argmax(-1)
            for r in range(B):
                out.append(tk.pre_tokenizer.decode([x for x in ids[r].tolist() if x not in (EOS, PAD)], skip_special_tokens=True))
    return out
def sample_stats(net):
    ts = sample_texts(net)
    Bs = [f["B"] for f in facts]; As = [f["A"] for f in facts]
    return dict(n=len(ts), B_mentions=sum(1 for t in ts if any(b in t for b in Bs)),
                A_mentions=sum(1 for t in ts if any(a in t for a in As)))

base_ctrl_ce = {c["i"]: ctrl_ce_all(net0, c) for c in CTRL}
PRE = dict(eff=fact_scores(net0, facts), gen=fact_scores(net0, genpairs), drift=0.0, samples=sample_stats(net0))
print("PRE", json.dumps({k: PRE[k] for k in ("eff", "gen")}), flush=True)
results = {"audit": dict(built=n_cand, codec_ok=len(keep)),
           "pre": PRE, "ctrl": dict(eff=fact_scores(net0, facts), gen=fact_scores(net0, genpairs),
                                    drift=ctrl_drift(net0, base_ctrl_ce), samples=sample_stats(net0))}
json.dump(results, open(OUTJ, "w"), indent=1, default=str)

# ---------- arms ----------
def roml_arm(net, fl):
    # trace best layer on first TR_NFACTS facts
    trace_loss = {l: [] for l in LAY_CAND}
    for f in fl[:TR_NFACTS]:
        j = f["Dlvl"]; p0 = int(torch.nonzero(f["D"][OFF[j]:OFF[j] + LEVELS[j]])[0])
        for l in LAY_CAND:
            v = torch.zeros(1, 1, 384, device=dev, requires_grad=True)
            ov = torch.optim.Adam([v], lr=0.05)
            for _ in range(TRACE_STEPS):
                ov.zero_grad(set_to_none=True)
                loss = ce_at(net, f["codes_B"], f["D"], j, site=(l, p0), patch=v)[0]
                if loss is None: break
                loss.backward(); ov.step()
            trace_loss[l].append(float(loss))
        lay = min(LAY_CAND, key=lambda l: sum(trace_loss[l]) / max(1, len(trace_loss[l])))
    print("  roml layer means", {l: round(sum(v) / max(1, len(v)), 3) for l, v in trace_loss.items()}, "chosen", lay, flush=True)
    covs = {}
    for fi, f in enumerate(fl):
        j = f["Dlvl"]; p0 = int(torch.nonzero(f["D"][OFF[j]:OFF[j] + LEVELS[j]])[0])
        if j not in covs:
            C = torch.zeros(1536, 1536, device=dev)
            for nd in NEIGH:
                Kk = level_keys(net, nd["codes"], j, lay)
                C += Kk.T.float() @ Kk.float()
            C /= max(1, NNEIGH * LEVELS[j])
            covs[j] = C + 0.1 * torch.diagonal(C).mean() * torch.eye(1536, device=dev)
        # v* trace
        v = torch.zeros(1, 1, 384, device=dev, requires_grad=True)
        ov = torch.optim.Adam([v], lr=0.05)
        for _ in range(TRACE_STEPS):
            ov.zero_grad(set_to_none=True)
            loss = ce_at(net, f["codes_B"], f["D"], j, site=(lay, p0), patch=v)[0]
            if loss is None: break
            loss.backward(); ov.step()
        v_star = v[0, 0].float()
        k_star = level_keys(net, f["codes_B"], j, lay)[p0].float()
        Lc = torch.linalg.cholesky(covs[j])
        Cinv_k = torch.cholesky_solve(k_star[:, None], Lc)[:, 0]
        a = Cinv_k / (1.0 + k_star @ Cinv_k)
        W = net.transformer.blocks[lay].mlp.c_proj.weight
        with torch.no_grad():
            W += ((v_star - W @ k_star)[:, None] @ a[None, :]).to(W.dtype)
        if fi % 50 == 0: print(f"  roml edited {fi}/{len(fl)} ({time.time()-t0:.0f}s)", flush=True)
    return lay

def lora_ft_arm(net, fl, mode):
    layers = LORA_LAYERS[:] if mode == "lora" else LAY_CAND[:]
    if mode == "lora":
        mods = {}
        for l in layers:
            A = (torch.randn(LORA_R, 1536, device=dev) * 0.01).requires_grad_(True)
            Bm = torch.zeros(384, LORA_R, device=dev).requires_grad_(True)
            mods[l] = (A, Bm)
            def mk(l):
                def hk(m, inp, out, l=l):
                    return out + F.linear(F.linear(inp[0], mods[l][0]), mods[l][1]) * LORA_SCALE
                return hk
            net.transformer.blocks[l].mlp.c_proj.register_forward_hook(mk(l))
        for p in net.parameters(): p.requires_grad_(False)
        params = [t for ab in mods.values() for t in ab]
        for t in params: t.requires_grad_(True)
    else:
        net.requires_grad_(True)
        params = list(net.parameters())
    opt = torch.optim.AdamW(params, lr=LORA_LR if mode == "lora" else 3e-5)
    steps = LORA_STEPS if mode == "lora" else FT_STEPS
    for s in range(steps):
        opt.zero_grad(set_to_none=True)
        sel = [fl[q] for q in torch.randperm(len(fl))[:16].tolist()]
        tot, cnt = 0.0, 0
        for f in sel:
            j = f["Dlvl"]
            loss = ce_at(net, f["codes_B"], f["D"], j)[0]
            if loss is None: continue
            (loss / 16).backward(); tot += float(loss); cnt += 1
        if LOC > 0 and mode == "lora":
            for c in [CTRL[q] for q in torch.randperm(len(CTRL))[:3].tolist()]:
                jc = torch.randint(1, K, (1,)).item()
                rl = ce_at(net, c["codes"], torch.ones(L, dtype=torch.bool, device=dev), jc)[0]
                if rl is not None: (LOC * rl / 3).backward()
        opt.step()
        if s % 100 == 0: print(f"  {mode} step {s} loss {tot/max(1,cnt):.3f} ({time.time()-t0:.0f}s)", flush=True)

def run_arm(name):
    net = copy.deepcopy(net0)
    for p in net.parameters(): p.requires_grad_(False)
    info = {}
    if name == "roml":
        info["layer"] = roml_arm(net, facts)
    elif name in ("lora", "ft"):
        train_fl = facts + (gviews[:GENVIEWS] if (GENVIEWS and name == "lora") else [])
        lora_ft_arm(net, train_fl, name)
    post = dict(eff=fact_scores(net, facts), gen=fact_scores(net, genpairs),
                drift=ctrl_drift(net, base_ctrl_ce), samples=sample_stats(net))
    post.update(info)
    results[name] = post
    print(name.upper(), json.dumps(post), flush=True)
    del net; torch.cuda.empty_cache()
    json.dump(results, open(OUTJ, "w"), indent=1, default=str)

for arm in ARMS: run_arm(arm)
json.dump(results, open(OUTJ, "w"), indent=1, default=str)
print("DONE", f"{time.time()-t0:.0f}s", flush=True)
