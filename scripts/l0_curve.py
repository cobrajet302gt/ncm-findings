import os, sys, glob, math, json
sys.path.insert(0, "/home/cobrajet/ncm-lab/ncm")
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["OMP_NUM_THREADS"] = "6"
import torch
torch.set_num_threads(6)
from omegaconf import OmegaConf
from tokenization.vqvae_tokenizer import VQVAETokenizer
CK = "/home/cobrajet/ncm-lab/artifacts/rustcodec-4k"
CFG = OmegaConf.load(f"{CK}/config.yaml")
CFG.tokenizer.pre_tokenizer.vocab_file_path = f"{CK}/bpe_vocab_8192.pickle"
CFG.tokenizer.pre_tokenizer.merges_file_path = f"{CK}/bpe_merges_8192.pickle"
segs = torch.load(f"{CK}/segments.pt", map_location="cpu", weights_only=True)
g = torch.Generator().manual_seed(0)
idx = torch.randint(0, segs.shape[0], (2048,), generator=g)
rows = []
for ck in sorted(glob.glob(f"{CK}/tokenizer*.pt"), key=lambda p: int(''.join(c for c in os.path.basename(p) if c.isdigit()) or "0")):
    CFG.tokenizer.vqvae.checkpoint_path = ck
    tk = VQVAETokenizer.from_config(CFG.tokenizer)
    tk.vqvae = tk.vqvae.to("cpu").eval()
    with torch.no_grad():
        codes = tk.encode_pre_tokenized_idx(segs[idx][::4])
    v = torch.bincount(codes[:, 0], minlength=int(CFG.tokenizer.vqvae.vqvae_vocab_size[0])).float()
    s = v.sum().item()
    nz = int((v > 0).sum()); eff = 1.0 / sum((x / s) ** 2 for x in v.tolist() if x > 0)
    H = -sum((x / s) * math.log(x / s, 2) for x in v.tolist() if x > 0)
    rows.append({"ck": os.path.basename(ck), "nonzero": nz, "eff": round(eff, 1), "H_bits": round(H, 2)})
    print(rows[-1], flush=True)
json.dump(rows, open("/home/cobrajet/ncm-lab/eval/rustcodec-l0-entropy-curve.json", "w"), indent=1)
print("DONE")
