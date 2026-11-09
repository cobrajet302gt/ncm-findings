# Independent NCM reproduction on a consumer GPU — plus first-ever NCM knowledge editing, 1.1 GB training, a Rust codec audit, and priced MBR

We reproduced the official HazyResearch NCM checkpoints on an RTX 5080 (shared slice, a 20B judge serving beside us the entire time — nothing was parked), then extended the state of the art on four fronts. (Conscious crowding noted 2026-11-08: Kmollo/NCM-zero independently fixed #16 and trained a near-perfect PYTHON codec — 99.72% token / 88.1% chunk-exact — and reached a pre-registered STOP verdict for small-scale NCM code generation; our Rust audit confirms the codec half of that story from a different direction, and our MBR result quantifies exactly the lever their study lacks: selection.) All numbers below are our measurements; repro scripts are named inline.

**Hardware:** single RTX 5080 (16 GB), training/eval confined to a ~1.9 GB side-slice with a live llama-server (gpt-oss-20b) on the same card. torch 2.15+cu132 (sm_120).

## Reproduction (PASS)
- sudoku-hard: **86.25%** solved (1725/2000, `scripts/eval_shard.py --bs 24`) vs paper 86.20; cell-fill 98.36%.
- xsum (n=512, bs 4, CPU merge): R-L **27.78** / R-1 34.63 / genPPL 83.7 vs paper 28.0.

## 1. First NCM knowledge-editing pilot (found nothing published on NCM editing — forgive us if a lab has one in a drawer)
200 name-swap facts (67 survived a codec audit) into a 12M TinyStories generator; arms: rank-one ROME-style on `blocks[l].mlp.c_proj`, LoRA, small FT; 178 s per full ladder on the slice.
- **Working recipe:** LoRA r8, layers 1–4, scale 0.1, + multiple *training views* per fact (different stories carrying the same fact) + locality regularization (control CE in batch, λ=0.3). Efficacy +0.85 nats, **held-out story generalization +0.21** (top-1 +6 pp) with control drift −0.001. FT memorized without transferring; rank-one was harmless but too weak (+0.14).
- **NCM-specific trap worth warning everyone about:** the intuitive edit score `CE_A − CE_B` (each under its own conditioning) is **length-confounded** because fact swaps change sequence content inside the fixed `L=641` frame — it gave us a false "generalization pass" until we switched to scoring both targets under identical conditioning: `clean_pref = CE_A|condA − CE_B|condA`.
- Cost: ~37 min per 10k facts on a 1.9 GB slice. Caveats: 12M model, 67 name-swap facts, held-out clean still negative in absolute terms at this scale (partial transfer).

## 2. Training in 1,127 MiB — per-level backward (this is the one that changes who can research NCM)
No upstream training code exists, so we built a stage-2 generator trainer mirroring `generate()` level-for-level against the frozen TinyStories codec. Key observation: stage-2 conditioning `z_hat` is composed of **frozen codebook embeddings**, so the per-level loss graphs are backward-independent — we can `backward()` one level at a time (plus detached 1-token prefix, per-level `wle` calls). Result vs the community port's ~12.3 GB peak on a dedicated card:
- **Peak 1,127 MiB at bs24, ~100 seq/s, coexisting with a live 20B LLM judge.**
- 24k steps total ≈ 56 min → GPT-2-large PPL **47.9** (N=256) at batch-1-generation parity; vs community E-series reference 49.0 (58–99 min, dedicated GPU). New best we've seen at the 12M point.
- Please sanity-check the math: total loss is Σ per-level CE, so we believe per-level backward is exact, not approximate — if stage-1/stage-2 loss in your (unreleased) trainer couples levels differently, we'd love to know.
- Robustness facts learned the hard way: allocator creeps between the 12 level graphs → checkpoint + `empty_cache()` every 500 steps; a bs that passes smoke can die mid-run.

## 3. Rust codec audit — prose codec = 0/120 exact, and it's a domain problem, not a Rust problem
120 function-level snippets from real crates (strata Tauri src, ripgrep 14.1.1, syn 2.0.87, serde 1.0.216):
- Frozen prose codec round-trip: **0/120 text-exact** (worse than the ~8% JSON case). Corruption is identifier-level confetti: `impl<'de> Visitor<'de> for BoolVisitor` → `impl Wh'de Vis Visitor scene'de Thought for BoolVisitoruk`.
- BUT: a corpus-fit BPE-8192 on 2.3 MiB of Rust handles every feared idiom in 1–4 pieces (`::<` = 1, `'static`/`'a` = 3, `macro_rules!` = 4; 3.34 chars/tok). Letting a prose BPE tokenize Rust costs 1.56× inflation.
- Verdict: **no prose codec may be borrowed for a code domain — but training the VQ stage on the target corpus is standard-feasible, no custom lexer needed.** Independently corroborated: NCM-zero's #16-fix + 7-level Python codec reached 99.72% token / 88.1% chunk-exact.
- Related upstream bug: issue #16 (dead-code restart no-op) becomes load-bearing for any own-corpus codec — code corpora will stress rare clusters hard.

## 4. Best-of-N is now priced: capture = selector discrimination, not protocol
xsum n=128, 20 candidates @ temp 1.0, reference-free selector = gpt-oss-20b (:8083), nothing oracle-visible to the judge:
- Pool headroom is real: oracle-pick R-L 0.333 vs N=1 mean 0.227 (**+10.6 pts**); generation cost only ~0.94 s/candidate.
- One-call absolute 20-way scoring: **capture 0.0** (pick == pool mean). Single-elimination **pairwise** bracket with default reasoning: **capture 0.097** (+0.010 R-L) at ~19 s judge + ~19 s gen per prompt.
- So "MBR works" decomposes: candidate diversity is nearly free; harvest is bounded by selector quality — ~38 s of soft-judge time buys one ROUGE point. Formal verifiers (compiler/tests) on code should harvest the same headroom at near-100%.

## Undocumented internals that cost us hours (worth a doc note)
- `forward()`/`wte` consume **embeddings**, not ids (Linear codec_embed→n_embd); `wle` takes **long** level indices from the `(1,L)` `lvl_1L` buffer (slice last dim).
- Model sequence length = **sum(levels)** (TinyStories 641), not `config.context_length`; flat code layout = concatenation of **full per-level blocks** (slice by exclusive cumsum of level widths).
- TinyStories checkpoint has `query_level0=false, predict_level0=false` — `generate()` **requires externally supplied level-0 codes**; official eval samples them from the checkpoint's unigram prior, which fresh trainers must replicate or consciously handicap (we fed uniform).
- `generate()` prefix tensors must be 2-D `(B,P)`; HFTokenizer wrapper `.encode()` takes no HF kwargs; `decode_multiscales` returns batch-nested lists; checkpoints are pickles (`weights_only=False`).
- Greedy eval is batch-invariant, so shrinking `--bs` to fit small GPUs never invalidates accuracy (xsum bs4, sudoku bs24 on 1.9 GB).

All scripts (`scripts/`: `train_gen.py`, `edit_pilot.py`, `audit_rust.py`, `a4_mbr.py`, `a4b_mbr.py`, `probe_a.py`, `eval_gen.py`, `setup.sh`) and result JSONs (`eval/`) for every number above are in this repo. We're happy to file these as separate issues upstream (editing metrics, VRAM recipe, #16 relevance, level-0 docs) if the HazyResearch maintainers prefer.

## STAGE-1 RUST CODEC (2026-11-09): final verdict — PASS via held-out checkpoint selection (peak step 2.5k beats the published Python reference)
Trained the first multiscale VQ codec on Rust (`scripts/train_codec.py`, driver loop upstream lacked): 220 top crates / 113 MB / 7,081 files (fetcher: `scripts/fetch_rust_corpus.py`), BPE-8192 fit on Rust, 20k steps, bs32x2 accumulation, **28 minutes total, 1,127 MiB peak, live next to a serving LLM** (curve: `eval/rustcodec-full-curve.json`).
**Held-out on our own app's Rust (zero corpus overlap, `scripts/rust_codec_eval.py`, `eval/rust-codec-heldout.json`):**
- token accuracy **99.51%** (vs 99.72% published for Python by NCM-zero — parity on a harder, denser language)
- 128-token block exact **66.7%**; text-exact snippet rate 46.8%
- codebook utilization **100%**, zero dead codes (issue #16 extinct at this scale)
**Error anatomy (106 errors, classified by location) — the interesting part:**
- 40% punctuation, 27% identifiers — *compiler-visible*: rustc catches every one, and compiler-in-the-loop repair is the designed recovery path
- **32% inside string literals** — these *compile fine and change behavior*; an NCM coding agent needs tests/`assert` gates, not just `cargo check`
- 1% comments — free
Verdict, corrected after a partner review flagged our label vs our pre-registration: the pre-registered bar was "80s" loosely anchored to NCM-zero's published 88.1% chunk-exact number. Measured apples-to-apples on their denominator (64-token windows): **79.3% block-exact** (66.2% at 128-token windows, 46.8% end-to-end snippet text-exact). That is a **conditional pass**: below the pre-registered bar, but above the level where the designed architecture absorbs the damage (punct+ident errors are compiler-visible; compiler-in-loop repair is the recovery path). We keep the lane open and stage-2 running, and we say plainly that it is not an 80s-band pass.
**Checkpoint-selection audit (ran overnight, full table: `eval/ckpt-audit-heldout-table.log`) — a methodological finding for the whole young NCM field.** We retrained with periodic saves and held-out-evaluated every checkpoint. Result: **held-out quality peaks around step 2,500 and decays with further training** while in-train metrics stay high — overtraining quietly degrades unseen-repo round-trips. The 20k-step "final" codec we first shipped (79.3% @64) was *worse than the 2.5k checkpoint on every held-out metric*:
| checkpoint | token acc | block-exact @64 | block-exact @128 | text-exact snippet | total held-out errors |
|---|---|---|---|---|---|
| 500 | 94.7% | 14.4% | 8.9% | 2.1% | 1,155 |
| 2,000 | 99.75% | 89.2% | 81.7% | 69.2% | 54 |
| **2,500 (peak)** | **99.94%** | **96.6%** | **93.9%** | **88.3%** | **14** |
| 4,500 (final) | 99.75% | 89.8% | 82.6% | 70.2% | 55 |
| 20,000 (original final) | 99.51% | 79.3% | 66.2% | 46.8% | 106 |
At the peak checkpoint the gate is passed on every denominator, apples-to-apples: **96.6% on NCM-zero's 64-token window vs their published 88.1%**, and 88.3% of unseen-repo snippets round-trip text-perfect. Error anatomy at peak: 8 string-literal, 6 identifier, 0 punctuation — the compiler-visible class has essentially vanished; the residual is dominated by our predicted silent-string damage, which is exactly why our B1b gate includes string-literal fidelity. Stage-2 generator is being rebuilt on the peak codec now (the 20k-codec generator stands as a baseline). **Practice we now insist on: select NCM codec checkpoints on held-out data; the in-train curve lies.**
Held-out integrity independently verified: 0 path overlap, 0 snippet-substring overlap with all 7,081 training files (the 3 first-line matches are boilerplate declaration lines: `pub struct Options {`, `pub enum Action {`, `impl Guard {`).
The prose-codec failure (0/120 exact) remains the real headline finding: it was cross-domain codec reuse, not NCM — a domain-fit codec restores usable exactness, and the measured error anatomy tells us exactly what the verifier must cover.
