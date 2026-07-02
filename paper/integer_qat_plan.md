# Integer-Only NanoChat via QAT — Setup, Training, and Determinism Experiments

This is the build-and-experiment plan for an **integer-only** NanoChat trained with
**quantization-aware training (QAT)**, and the full determinism test suite (batch
invariance, cross-platform, Feynman-1000, fidelity) used to evaluate it.

The scientific claim this enables — which the current PTQ W8A8-linear prototype
**cannot** support — is:

> An integer-only Transformer computes its logits with no non-associative
> floating-point reductions and no platform-specific transcendental libraries, so
> the int32 logits (and therefore the greedy argmax) are **bitwise identical
> across batch shapes, kernels, and hardware platforms.**

The measurable target: the SHA-256 hash of the int32 logit tensor is identical for
batch 1/8/64 and across GPU/CPU, while FP32/FP16/BF16 and PTQ-W8A8 diverge.

---

## 0. Prerequisites and honest scoping

- **QAT is training. It needs a GPU.** Two viable budgets are given in §4; the
  cheap path (QAT-finetune the existing `paper-d12` checkpoint) is a few GPU-hours,
  not a from-scratch run.
- **Train/inference parity is the #1 correctness rule.** The fake-quant forward
  used in training must be *arithmetically equivalent* to the integer inference
  path. The model you train must be exactly the model you deploy, or the
  determinism result is meaningless. Every phase below has a parity check.
- Reuse everything already built: [scripts/repro_experiment.py](../scripts/repro_experiment.py),
  [scripts/feynman_1000.py](../scripts/feynman_1000.py),
  [runs/reproducibility.sh](../runs/reproducibility.sh),
  [runs/quality_eval.sh](../runs/quality_eval.sh). The QAT model is added as one
  new `precision` value: `int-qat`.

---

## 1. What changes vs the float model (op-by-op integer spec)

Every floating-point op in [nanochat/gpt.py](../nanochat/gpt.py) needs an integer
replacement. Bit-widths are chosen for **determinism, not compression** — the big
matmuls are INT8 (the "quantization" story), everything else is wide INT
fixed-point to preserve accuracy.

| Op | Location | Float today | Integer target | Width |
|---|---|---|---|---|
| Core linears (q,k,v,proj,fc,proj,lm_head) | gpt.py:75-78,132-133,175 | fp32/bf16 matmul | INT8×INT8→INT32 + **dyadic requant** | w:int8 (per-channel), a:int8 (per-token), acc:int32 |
| RMSNorm (no-param) | gpt.py:42 | `F.rms_norm` (bf16) | integer inverse-sqrt (Newton) | int32 internal, int16 out |
| Attention QKᵀ / ×V | flash_attn, gpt.py:108 | fp | integer matmul + requant | int8 in, int32 acc |
| **Softmax** | inside flash_attn | fp exp/÷ | **I-BERT integer exp** (2nd-order poly) + int normalize | int32 fixed-point |
| Rotary `x1·cos+x2·sin` | gpt.py:57-63 | fp cos/sin | fixed-point cos/sin tables | int16 (Q15) |
| relu² | gpt.py:137 | fp | integer square (exact) | int32 |
| sigmoid gates (smear, ve_gate) | gpt.py:94,436-449 | fp sigmoid | integer sigmoid LUT | int16 |
| resid/x0/backout/smear λ scalars | gpt.py:457,464 | fp mul | fixed-point scalar (dyadic) | int32 |
| Embeddings (wte, value_embeds) | gpt.py:172,190 | fp table | quantized int8 table + scale | int8 |
| softcap `15·tanh(z/15)` | gpt.py:472 | fp tanh | **skip for greedy** (monotonic ⇒ argmax unchanged); LUT if sampling | — |
| Final logits | gpt.py:469 | fp | **int32**, argmax directly | int32 |

Notes:
- **relu² is exact** in integer (square of an int) — free.
- **max reductions are order-independent**, so per-token activation max (for
  dynamic scales) is already deterministic across platforms; only the *divide* by
  the scale must be made integer (dyadic).
- **softcap tanh is monotonic** ⇒ it never changes a greedy argmax; drop it in the
  integer greedy path and note it in limitations. Keep a LUT version only if
  running the sampling experiments.

---

## 2. QAT methodology

### 2.1 Fake-quant + straight-through estimator (STE)
During training, simulate integer rounding in the forward pass and pass gradients
straight through the round:
```
x_q = x + (quant(x) - x).detach()   # forward = quantized, backward = identity
```
Master weights stay fp32 for the optimizer (Muon/AdamW as configured in
[gpt.py:374](../nanochat/gpt.py#L374)); only the forward is fake-quantized.

### 2.2 Quantization config (recommended defaults)
- **Weights:** symmetric, **per-output-channel**, int8. (Matches
  [static_quant.py:63](../nanochat/static_quant.py#L63).)
- **Linear activations:** symmetric, **per-token** (per-row) dynamic, int8. Per-token
  fixes the outlier problem that made the PTQ per-tensor version degrade.
- **Nonlinear internals** (norm, softmax, rotary): fake-quant to the *same*
  fixed-point widths the integer inference path uses (int16/int32), so parity holds.
- **Embeddings:** per-channel int8.

### 2.3 Progressive fake-quant schedule (reduces training instability)
Turn on fake-quant in stages so the model adapts gradually:
1. Steps 0–20%: weights fake-quant only.
2. 20–50%: + linear activations.
3. 50–100%: + norm/softmax/rotary fixed-point.

Optionally warm-start weights from the existing `paper-d12` checkpoint (§4, cheap
path) so stages 1–2 converge fast.

---

## 3. Code to build

Create these modules (keep them small and independently tested):

- `nanochat/int_ops.py` — integer primitives + their float references and unit
  tests: `dyadic(scale)`, `apply_dyadic`, `int_isqrt`, `int_rmsnorm`, `int_relu2`,
  `int_exp` (I-BERT), `int_softmax`, `int_sigmoid`, `fixed_rotary`.
- `nanochat/qat.py` — `FakeQuant` modules + `quantize_model_qat(model, cfg)` that
  wraps the float model for training. Records activation ranges/observers.
- `nanochat/int_gpt.py` — `IntGPT`, the **true integer** inference path mirroring
  `qat.py` op-for-op. Loads a QAT checkpoint, converts to integer buffers, and runs
  a forward that returns **int32 logits**.
- `tests/test_int_ops.py` — each primitive vs its float ref within tolerance.
- `tests/test_parity.py` — `IntGPT` logits == QAT fake-quant logits (the parity gate).

Integration points:
- [scripts/base_train.py](../scripts/base_train.py): add `--qat` (and `--qat-init-from`
  to warm-start from `paper-d12`). When set, call `quantize_model_qat` after
  `init_weights`/load.
- [scripts/repro_experiment.py](../scripts/repro_experiment.py) and
  [scripts/feynman_1000.py](../scripts/feynman_1000.py): add `int-qat` to the
  `--precision` choices; when selected, load via `IntGPT` and skip float dtype
  validation. Add a `--emit-logit-hash` flag that writes `sha256(int32_logits)`.

---

## 4. Compute budget & training recipe

The paper model is depth-12, ~286M params, trained on 1.321B tokens (D:N≈12).

### Cheap path (recommended first): QAT-finetune from `paper-d12`
Warm-start from the existing checkpoint and finetune with fake-quant for ~10–20%
of original tokens (~150–250M tokens). Expect a few GPU-hours on a single
A100/H100 (rentable).
```bash
python -m scripts.base_train \
  --qat --qat-init-from paper-d12 --qat-init-step 2520 \
  --depth 12 --aspect-ratio 64 --head-dim 128 --window-pattern SSSL \
  --num-iterations <~10-20% of original> \
  --eval-every 100 --model-tag paper-d12-qat
```

### From-scratch path (only if finetune underperforms)
Full pretrain with the progressive schedule (§2.3). Roughly the original
`paper-d12` cost; multi-GPU. Use [runs/speedrun.sh](../runs/speedrun.sh) as the
template, adding `--qat`.

**Accuracy gate:** QAT val BPB should land within ~0.01–0.02 of the fp32 baseline
(0.8395). If it blows past that, revisit per-token activations / softmax precision
before scaling training.

---

## 5. Integer-only inference path (deploy)

`IntGPT` (`nanochat/int_gpt.py`) runs entirely in integer arithmetic:
1. Quantize embeddings offline to int8 + per-channel scales.
2. Forward: dyadic requant between all linears; integer RMSNorm; fixed-point
   rotary; integer sigmoid gates; integer QKᵀ → I-BERT integer softmax → ×V;
   integer relu² MLP; integer residual with dyadic λ.
3. Output **int32 logits**; `argmax` over int32.

**Parity gate (must pass before any experiment):**
```
max |IntGPT.logits_dequant - QAT.fakequant.logits| == 0   (or ≤ 1 ULP of the shared scale)
```
Run on a handful of prompts. If parity fails, the determinism claim is invalid.

---

## 6. Determinism validation

Three levels, strongest first:

1. **Bitwise logit determinism (the headline).** Emit `sha256(int32_logits)` for
   each prompt/token under every condition. For `int-qat` the hash must be
   **identical** across batch 1/8/64 and across GPU/CPU. For fp32/fp16/bf16/PTQ it
   will differ. This is the definitive result.
2. **Behavioral (token) determinism.** Reuse the existing metrics: unique
   sequences, FP32-reference match rate, modal fraction, first-divergence token
   index.
3. **Numerical drift** (for the float baselines only): logit |Δ|, RMSE, KL — shows
   *why* the floats diverge and the integer path does not.

---

## 7. Experiment matrix

Conditions (all greedy, primary):

| Precision | Batch | Gens/cond | Platforms | Purpose |
|---|---|---|---|---|
| fp32 | 1,8,64 | 1000 | GPU, CPU | float baseline |
| fp16 | 1,8,64 | 1000 | GPU, CPU | float baseline |
| bf16 | 1,8,64 | 1000 | GPU, CPU | float baseline |
| ptq-w8a8 (static) | 1,8,64 | 1000 | GPU, CPU | your current prototype (control) |
| **int-qat** | 1,8,64 | 1000 | GPU, CPU (+2nd GPU if available) | the contribution |

Run via the existing harness (add `int-qat` to conditions in
[runs/reproducibility.sh](../runs/reproducibility.sh)):
```bash
MODEL_TAG=paper-d12-qat STEP=<step> TRIALS=5 REPLICAS_PER_PROMPT=10 \
  MAX_TOKENS=32 DECODE=greedy OUT=repro_results/qat_greedy \
  bash runs/reproducibility.sh
# repeat with DEVICE=cpu for the cross-platform arm
```

Metrics per condition: logit-hash agreement (batch & platform), reference-match
rate, unique sequences, modal fraction, val BPB, throughput.

**Primary hypotheses**
- **H1 (bitwise batch invariance):** `int-qat` logit hashes are identical across
  batch 1/8/64; floats are not (or only FP32 is, at token level).
- **H2 (cross-platform determinism):** `int-qat` logit hashes are identical
  GPU↔CPU; fp32=100% token match but *not* bit-identical; fp16/bf16/PTQ diverge
  (recall your PTQ result: 85/25% token match).
- **H3 (fidelity retained):** `int-qat` val BPB within ~0.02 of fp32, i.e. far
  better than the PTQ per-tensor prototype (0.875).

---

## 8. Feynman-1000 end-to-end test

Use [scripts/feynman_1000.py](../scripts/feynman_1000.py) (already implements
1,000×1,000-token greedy completions, token hashing, first-divergence). Add
`int-qat` to its `--precision` choices.
```bash
for BATCH in 1 8 64; do
  for DEV in cuda cpu; do
    python -m scripts.feynman_1000 --precision int-qat --batch-size $BATCH \
      --completions 1000 --max-tokens 1000 --device-type $DEV \
      --model-tag paper-d12-qat --step <step> \
      --reference-artifact repro_results/feynman_qat/reference.pt \
      --output repro_results/feynman_qat/int-qat_b${BATCH}_${DEV}.json
  done
done
```
Report: unique sequences, modal fraction, earliest divergent token, and
**cross-batch / cross-platform logit-hash agreement** — the Thinking-Machines-style
headline number.

---

## 9. Quality / fidelity

Reuse [runs/quality_eval.sh](../runs/quality_eval.sh); add `int-qat`. Report val BPB
over 2,097,152 tokens vs fp32/fp16/bf16 and the PTQ prototype. This is the
"is the deterministic model still good?" check (guards against deterministic-garbage).

---

## 10. Milestones & accuracy gates

| Phase | Deliverable | Gate |
|---|---|---|
| P0 | `int_ops.py` primitives + `tests/test_int_ops.py` | each primitive matches float ref within tol |
| P1 | integer MLP+linears+requant in `int_gpt.py` | partial-forward parity with fake-quant |
| P2 | integer RMSNorm + rotary + gates | parity holds; BPB (PTQ-style) sane |
| P3 | integer attention (QKᵀ, I-BERT softmax, ×V, GQA, window, KV cache) | full parity gate (§5) |
| P4 | `qat.py` + `--qat` training; finetune `paper-d12-qat` | val BPB within ~0.02 of fp32 |
| P5 | run full matrix + Feynman-1000 (§7–8) | int-qat logit hashes identical across batch & platform |
| P6 | update paper: positive cross-platform determinism result | — |

Stop-and-reassess triggers: parity fails (P1–P3), or QAT BPB > ~0.87 (P4).

---

## 11. Reproducibility artifacts

Archive with every run (the harness already captures most): checkpoint + tokenizer
SHA-256, QAT config JSON (bit-widths, per-op schedule), git commit + dirty state,
package freeze, `nvidia-smi -q` / CPU `platform.txt`, prompt + calibration
manifests, and the **int32 logit hashes** per condition.

---

## 12. Risks & mitigations

- **QAT unstable / BPB blows up** → progressive schedule (§2.3), warm-start from
  `paper-d12`, keep softmax/norm at int16+ precision, per-token activations.
- **Integer softmax hurts accuracy** → widen I-BERT poly precision; keep attention
  probs at int16.
- **Parity drift (train ≠ inference)** → the parity gate (§5) is mandatory before
  experiments; unit-test each op.
- **No second GPU for cross-platform** → CPU is a fully valid second platform
  (different kernels entirely); the logit-hash test is the same.
- **Deterministic-but-bad model** → the quality gate (§9) blocks this; report BPB
  alongside every determinism number.
