# Static Quantization as a Reproducibility Primitive for Transformer Inference

## Claim discipline

This is a position paper with a NanoChat prototype, not a completed integer-only
Transformer. The prototype uses frozen W8A8 quantization grids and PyTorch's
existing INT8×INT8→INT32 linear operation, then dequantizes for floating-point
nonlinear and residual operations. Use **static W8A8 linear** in experiment
labels. The evidence can support the claim that static
quantization reduces sensitivity to numerical perturbations; it cannot yet
support claims of cross-platform bitwise determinism or integer-only execution.

## Eight-page allocation

1. **Introduction and contribution — 0.75 page**
   - Floating-point inference is usually stable enough for deployment but does
     not promise bitwise equality across batch shapes, backends, or platforms.
   - Position: quantization should be studied as a reproducibility mechanism,
     not only a compression/throughput mechanism.
   - Contributions: motivation, prior-art synthesis, NanoChat W8A8 prototype,
     and controlled measurements with deterministic and dynamic controls.

2. **Motivation and problem statement — 1.0 page**
   - Define determinism, repeatability, reproducibility, and semantic stability.
   - Explain non-associativity and batch-dependent reduction/kernel choices.
   - State the research question: does a fixed quantization grid reduce output
     sensitivity relative to FP32 and FP16 under batch-shape perturbations?
   - Do not equate fixed random seeds with deterministic inference.

3. **Prior art — 1.25 pages**
   - Integer-only nonlinear approximations and dyadic arithmetic: I-BERT.
   - Integer-only attention/normalization work such as I-ViT.
   - LLM post-training quantization and W8A8 efficiency literature.
   - Cross-platform consistent post-training quantization.
   - Framework reproducibility guidance and known batch-versus-slice numerical
     differences.
   - Gap: most quantization work optimizes accuracy, memory, and latency rather
     than measuring generation reproducibility as the primary outcome.

4. **NanoChat prototype — 1.25 pages**
   - Model and checkpoint identification.
   - FP32: literal FP32 with TF32 disabled.
   - FP16: NanoChat's explicit float16 compute path.
   - BF16: native GH200 inference format.
   - Static INT8 prototype: symmetric INT8; per-output-channel weights,
     per-layer activation scale; eight fixed calibration strings;
     INT8×INT8→INT32 linear execution using PyTorch; no custom kernels.
   - Explain why this isolates grid snapping but is not integer-only inference.
   - Dynamic W8A8 control: identical integer matmul with runtime activation
     scaling, separating integer arithmetic from fixed-scale effects.

5. **Experimental protocol — 1.0 page**
   - Batch sizes 1, 8, 64; exactly 1,000 generations per condition:
     20 preregistered prompts × 10 replicas × 5 fresh process launches.
   - Identical prompt, checkpoint, maximum length, and decoding schedule.
   - Greedy primary experiment.
   - Fixed-sample secondary experiment uses pre-generated CPU uniforms shared
     by every replica, avoiding batch-dependent RNG consumption.
   - FP32 batch-1 teacher-forced trace is the common reference.
   - Deterministic FP32/FP16 controls use PyTorch deterministic algorithms and
     a fixed cuBLAS workspace configuration.
   - Held-out validation BPB measures fidelity; calibration sizes 1/8/32 form
     a prespecified static-scale sensitivity analysis.
   - Every launch uses the declared physical batch shape; padded rows are
     discarded rather than silently changing the final batch size.
   - Hardware/software details, package versions, Git state, prompt manifest,
     and `nvidia-smi` output are archived with the results.

6. **Results — 1.25 pages**
   - Table: unique sequences, reference-match rate, mean/max absolute logit
     difference, RMSE, mean/max KL divergence, and throughput.
   - Figure 1: reference-match rate by precision and batch size.
   - Figure 2: log-scale mean KL divergence by precision and batch size.
   - Figure 3: maximum absolute logit error by precision and batch size.
   - Report 95% prompt-cluster bootstrap confidence intervals. Replicated rows
     and process launches are repeated measurements, not independent prompts.

7. **Discussion, threats, and roadmap — 1.0 page**
   - A null result is meaningful: same-machine inference may already be stable.
   - W8A8 applies only to linear layers; nonlinearities, residuals, rotary
     embeddings, attention softmax, and dequantization remain floating point.
   - One prompt/checkpoint/device cannot establish generality.
   - Batch size changes tensor shapes but does not cover software/hardware drift.
   - Quantization error may exceed the floating-point differences it suppresses.
   - Roadmap: real quantized operators, integer softmax/RMSNorm, fixed rotary
     tables, cross-GPU/CPU replication, prompt suite, and quality evaluation.

8. **Conclusion — 0.5 page**
   - State only what the measurements show.
   - Frame full integer-only NanoChat inference as follow-on work.

## Primary hypotheses

- H1: FP16 has larger logit deviation from FP32 batch 1 than FP32 at batch 8/64.
- H2: static W8A8 linear inference has lower *within-condition output multiplicity*
  than FP16 under the same batch perturbations.
- H3: static W8A8 may improve reference-match rate while increasing
  KL divergence from FP32. This tradeoff separates reproducibility from fidelity.
- H4: static W8A8 is more stable than dynamic W8A8 if scale freezing, rather
  than integer matmul alone, is the relevant reproducibility mechanism.

H2 is the central hypothesis. H1 is a sanity check. H3 prevents a misleading
conclusion in which a stable but badly distorted model is called better.

## Required result matrix

| Precision | Batch | Generations | Decode |
|---|---:|---:|---|
| FP32 | 1, 8, 64 | 1,000 each | greedy |
| FP16 | 1, 8, 64 | 1,000 each | greedy |
| BF16 | 1, 8, 64 | 1,000 each | greedy |
| Deterministic FP32 | 1, 8, 64 | 1,000 each | greedy |
| Deterministic FP16 | 1, 8, 64 | 1,000 each | greedy |
| Static W8A8 linear | 1, 8, 64 | 1,000 each | greedy |
| Dynamic W8A8 linear | 1, 8, 64 | 1,000 each | greedy |

Repeat the complete matrix with `fixed-sample` as a secondary experiment. The
prompt manifest is fixed in `paper/prompts.jsonl` and must be archived unchanged.

## Prior-art starting points

- Kim et al., [I-BERT: Integer-only BERT Quantization](https://arxiv.org/abs/2101.01321).
- Li and Gu, [I-ViT: Integer-only Quantization for Efficient Vision Transformer Inference](https://arxiv.org/abs/2207.01405).
- [Post-Training Quantization for Cross-Platform](https://arxiv.org/abs/2202.07513).
- PyTorch, [Numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html).
- PyTorch, [Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html).

## Execution

Primary conference experiment:

```bash
MODEL_TAG=paper-d12 STEP=2520 TRIALS=5 REPLICAS_PER_PROMPT=10 MAX_TOKENS=32 DECODE=greedy \
  OUT=repro_results/greedy bash runs/reproducibility.sh
```

Secondary experiment:

```bash
MODEL_TAG=paper-d12 STEP=2520 TRIALS=5 REPLICAS_PER_PROMPT=10 MAX_TOKENS=32 DECODE=fixed-sample \
  OUT=repro_results/fixed_sample bash runs/reproducibility.sh
```

Held-out quality and calibration ablation:

```bash
MODEL_TAG=paper-d12 STEP=2520 OUT=repro_results/quality \
  bash runs/quality_eval.sh
```
