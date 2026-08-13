# DSA Decode Online Softmax Comparison

This example isolates the V3.2 decode online-softmax stage for FP32 scores
`[512, 2048]`, INT32 indices `[4, 2048]`, scale `1 / 24`, and input range
`[-24, 24]`. Both one-launch programs emit the internal BF16 numerator,
FP32 running max, FP32 running sum, and online-update old scale state; there is no
final device normalization pass. The vLLM no-update VF deliberately leaves
tile 0 scale unwritten and emits tiles 1--15, whereas SIMT initialization writes
tile 0 as zero before its per-tile VF updates. A NaN sentinel verifies that
distinction without changing either production device path.

The fixed decode shape maps 512 rows to 16 active Softmax row owners, with 32
rows per owner. The vLLM path uses the actual Arch35 aligned-128 no-update and
update VF bodies, while SIMT v2 preserves the production outer-loop boundary of
one `head64_fused_vllm_softmax_vf` call per 128-token tile.

These 16 blocks are the effective Softmax AIV owners, not necessarily the
physical MIX launch capacity of the enclosing attention operator. In the
vLLM-Ascend production kernel, the profiler reports 32 AIC / 64 AIV capacity,
but fixed-shape scheduling computes eight active AICs and their 16 AIVs. The
remaining physical tasks exit before this stage. SIMT v2 launches eight AICs /
16 AIVs for the same fixed shape.

Run on device 0 with:

```bash
RUN_ID=<unique-id> scripts/run.sh
```

The script targets `dav-3510`, runs correctness before profiling, retains five
raw `msopprof` trees for each target, and strictly parses target-kernel
`Task Duration`. It rejects missing or extra rows, launch-dimension mismatch,
and measured frequency mismatch. It writes `summary.csv` containing all five
samples plus median, minimum, and maximum, and `ratio.txt` containing the
vLLM-Ascend/SIMT-v2 ratio.

## Measured evidence

Collected on 2026-08-13 on the task-specified Ascend 950 device 0 at
`root@121.41.199.170:20002`. Both targets used `dav-3510`, CANN 9.2.0,
Bisheng clang 15.0.5 (`clang-5c68a1cb1231 flang-5c68a1cb1231`), and
`msopprof` 26.2.0. Each accepted row reports current/rated `1650/1650 MHz` and
block dim 16.

Both direct correctness runs passed. Each reported maximum reconstructed
probability error `2.28256e-06` and maximum row-sum error `0.00013417`.

| Implementation | Five Task Duration samples (us) | Median (us) | Minimum (us) | Maximum (us) |
| --- | --- | ---: | ---: | ---: |
| vLLM-Ascend | 17.766001, 17.965000, 17.485001, 18.098000, 17.476999 | 17.766001 | 17.476999 | 18.098000 |
| SIMT v2 | 59.422001, 59.452000, 59.327000, 59.631001, 59.486000 | 59.452000 | 59.327000 | 59.631001 |

The median `vLLM-Ascend / SIMT v2` ratio is `0.298829`; equivalently, SIMT v2
takes `3.3464x` the vLLM-Ascend time for this isolated one-kernel boundary.
The result uses five independent `msopprof` processes per implementation,
default profiler warmup (reported as five), exact kernel-name selection, and
only `Task Duration`.

- Remote raw root:
  `/tmp/dsa-decode-softmax-actual-evidence-20260813-2130`
- Retained local root:
  `/tmp/dsa-decode-softmax-actual-evidence-20260813-2130` (267 files, including
  ten raw OPPROF trees, parser output, correctness logs, and binaries)
- Standalone source hashes: vLLM-Ascend
  `cc17ae42dd5ca0a320f93a97e1cee131848f8ac59baafd8bf27ebe883d068845`;
  SIMT v2
  `24d607a1a9db288937d4f6143f0a224626005a89eb1117369a2c5c7fd54dbd76`
- Executable hashes: vLLM-Ascend
  `f40dddf3d63e8be5adc1fbc467bd13347dcbc253ce466d3f0646833c28a5b079`;
  SIMT v2
  `f1a930d5e0e0b056f4e4a7b9e5031da35eddbb9eb01b2f52a00e9f22b5753838`

The supplied V3.2 workflow BasicInfo profiles provide a boundary check. SIMT's
complete fused attention kernel is `76.810997` and `77.406998 us` with block
dim 16. The vLLM-Ascend complete SparseFlashAttention kernel is `55.884998`
and `56.605999 us` with physical MIX launch capacity 32 AIC / 64 AIV. Those
rows include QK, KV gather, Softmax, PV, output update, and synchronization, so
their ratio is not a Softmax ratio. They do confirm the expected fixed-shape
dispatch and show that the isolated SIMT Softmax boundary is a major portion of
its fused kernel.

These measurements are device task time and do not include host dispatch
latency.
