# Frozen Experimental Protocol

## Research question

Does calibration-fixed W8A8 linear inference reduce NanoChat's numerical and
generation sensitivity to batch shape and process restart relative to floating
point, and is any reduction attributable to fixed scales rather than integer
matrix multiplication alone?

## Frozen artifacts

- Checkpoint: `paper-d12/model_002520.pt`
- Checkpoint SHA-256: `611dd34d3be622e2aad7002c70de8994860e59c870c3a0065038f6c62bbbb086`
- Total parameters: 286,261,730
- Training: 1.321B tokens, BF16 compute, validation BPB 0.8395688
- Evaluation prompts: `paper/prompts.jsonl`
- Calibration inputs: `paper/calibration.jsonl`; no overlap by exact string

## Main factors

- Arithmetic method: FP32, deterministic FP32, FP16, deterministic FP16,
  BF16, static W8A8 linear, dynamic W8A8 linear.
- Physical batch size: 1, 8, 64.
- Decoding: greedy and fixed-sample.
- Repetition: 20 prompts × 10 replicas × 5 fresh processes = 1,000 retained
  generations for every method/batch/decoding condition.

The final partial logical batch is padded to the declared physical batch size;
padding rows are excluded from statistics.

## Feynman-1000 end-to-end test

Following Thinking Machines Lab, generate 1,000 greedy completions of exactly
1,000 tokens from the prompt `Tell me about Richard Feynman`. Run the protocol
at physical batch sizes 1, 8, and 64 for FP32, BF16, static W8A8, and dynamic
W8A8; deterministic FP and FP16 conditions may be added to the complete matrix.
Report unique sequences, modal fraction, FP32 batch-1 reference-match rate,
earliest divergent token, and outputs pooled across batch shapes.

## Outcomes

Primary generation outcomes:

1. Unique token sequences per prompt across process restarts.
2. FP32 batch-1 reference-match rate.
3. Modal output fraction.

Primary numerical outcomes:

1. Mean and maximum absolute logit difference.
2. Logit RMSE.
3. Forward KL divergence from the FP32 batch-1 distribution.

Fidelity and secondary outcomes:

1. Held-out validation bits per byte over 2,097,152 tokens.
2. Throughput.
3. Static calibration-size sensitivity at 1, 8, and 32 samples.

## Statistical analysis

- Report prompt-level means and 95% prompt-cluster bootstrap intervals using
  10,000 resamples and seed 20260621.
- Process repetitions and duplicate rows are repeated measurements, not
  independent prompts.
- Report every prespecified method and batch condition, including null results.
- Do not claim cross-platform determinism from a single hardware platform.

## Hardware replication

The complete frozen matrix should be run independently on GH200 and A100 or
H100. Results directories must retain the checkpoint/tokenizer hashes, Git
commit and dirty state, prompt/calibration manifests, package versions, and
`nvidia-smi -q` output.

## Claim boundary

The prototype is not an integer-only Transformer. Aligned core linear layers
use INT8×INT8→INT32 matrix multiplication; NanoChat's tiny unaligned smear and
value-gate projections remain floating point. Normalization, rotary embeddings,
attention softmax, residual operations, and dequantization remain floating
point. The work evaluates static quantization as a reproducibility primitive;
it does not propose a new quantization algorithm.

PyTorch's CUDA `torch._int_mm` rejects inputs with 16 or fewer rows. Both W8A8
conditions therefore append zero rows to a height of 32 for smaller decode
matmuls and slice those rows from the INT32 output. This shared compatibility
path is included in performance and limitation reporting.
