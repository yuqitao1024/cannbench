# DSA Decode Online Softmax Comparison Contract

This standalone example compares the online-softmax code actually called by the
vLLM-Ascend Arch35 Sparse Flash Attention path and CannBench SIMT v2 V3.2 decode.
It is a fixed performance experiment, not a reusable operator.

## Workload and output

- FP32 scores: `[512, 2048]`, deterministically bounded to `[-24, 24]`.
- INT32 selected indices: `[4, 2048]`, including invalid and V3.2 causal cases.
- Scale: `1 / 24`; tile size: 128 selected tokens; target: `dav-3510`.
- The fixed shape has 16 active Softmax row owners. Each owns 32 rows, so both
  standalone kernels launch with block dim 16.
- One kernel launch emits the original internal BF16 numerator in NZ layout,
  FP32 running max, FP32 running sum, and the online-update `old scale` state. The
  vLLM no-update VF leaves tile 0 unwritten and emits update scales for tiles
  1--15; SIMT initialization writes tile 0 as zero and its VF writes tiles
  1--15.
- The vLLM VF keeps compile-time `s1BaseSize = 64` separate from the 32 active
  rows: it writes a `33 * 128` staging tensor, then uses the upstream eight-block
  2D copy semantics to form the dense 64-row NZ output shared with SIMT.
- There is deliberately no final device normalization kernel. The host
  reconstructs final probabilities by applying subsequent old-scale factors.

The host constructs the production stage inputs directly. It turns invalid or
causally masked vLLM score entries into `-inf`, matching the score-tile contract
at the Arch35 `ProcessVec1` boundary. The SIMT v2 VF retains its production
indices lookup and causal predicate. Both paths preserve 128-token online
updates, BF16 conversion, and separate kernel symbols. The ACL Runtime host
launches device 0, synchronizes, copies outputs, and checks a stable FP32 oracle
in the same tile order.

The production call structure is part of the contract. vLLM executes the
aligned-128 no-update VF on tile zero and the aligned-128 update plus
`SFAUpdateExpSumAndExpMax` VFs on subsequent tiles. SIMT initializes UB state
once, then its outer AIV loop calls `head64_fused_vllm_softmax_vf` once for each
128-token tile; the VF itself does not own the 16-tile loop. The host initializes
the device scale output to NaN and verifies this write boundary explicitly; it
does not add a tile-zero store to the migrated vLLM kernel.

## Acceptance thresholds

- Running max: absolute/relative `2e-5 / 2e-5`.
- Running sum, update scales for tiles 1--15, and SIMT tile-zero scale:
  absolute/relative `2e-3 / 2e-3`. vLLM tile zero must remain the NaN sentinel.
- BF16 numerator: absolute/relative `8e-3 / 8e-3`.
- Reconstructed probability: absolute/relative `2e-4 / 2e-3`.
- Each finite reconstructed row sum has absolute error at most `2e-3`.

## Measurement

Correctness must pass before profiling. Each target then receives five
independent `msopprof` collections. Each raw tree must contain exactly one
exact-name target row with block dimension 16 and a valid measured frequency.
Only `Task Duration` is reported. Raw artifacts remain alongside structured
samples, median, minimum, maximum, and vLLM-Ascend/SIMT-v2 ratio.
