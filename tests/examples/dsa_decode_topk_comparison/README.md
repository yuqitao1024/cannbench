# DSA Decode TopK comparison

This standalone Ascend 950 experiment compares vLLM-Ascend's Arch35 BF16 16K
trunk TopK against the distributed TopK selected by CannBench SIMT v2 for V3.2
decode. Both map BF16 scores [4, 32768] to INT32 indices [4, 2048] with one
kernel launch and use only ACL Runtime on the host. Output ordering is
intentionally not compared, and equal-score index identity is intentionally not
compared.

Build and collect with:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
flock -x /tmp/cannbench-dsa-stage-comparison.lock bash -lc 'RESULT_ROOT="$PWD/evidence/<unique-run>" bash scripts/run.sh'
```

The script configures `dav-3510`, verifies both executables first, then creates
five independent `msopprof` `Default` collections per implementation. It uses
only the single target row's `Task Duration`, requires launch-count one,
production block dimensions (vLLM-Ascend 4, SIMT v2 64), and frequency parity,
and retains every raw collection below `evidence/`.

## Measured evidence

Collected on 2026-08-13 on the task-specified Ascend 950 device 0 at
`root@121.41.199.170:20002`. The target was compiled for `dav-3510` with CANN
9.2.0 and Bisheng clang 15.0.5
(`clang-5c68a1cb1231 flang-5c68a1cb1231`). The profiler was `msopprof` 26.2.0.

- Profiler mode: `Default`, default profiler warmup (reported as 5), exactly
  one application launch selected per independent collection
- Frequency: every selected row reports current/rated `1650/1650 MHz`
- Production launch ownership: vLLM-Ascend 4 blocks; SIMT v2 64 blocks
- Remote root:
  `/tmp/cannbench-dsa-topk-actual-20260813-200615/dsa_decode_topk_comparison/evidence/actual-distributed-20260813-200615`
- Retained local root:
  `/tmp/dsa-decode-topk-actual-evidence-20260813-200615/actual-distributed-20260813-200615`
  (262 files, including 10 raw OPPROF trees, parser output, correctness logs,
  and build logs). Raw profiler trees remain outside the repository.
- Standalone source hashes: SIMT v2
  `5ee3ff8559b824866da939c9cff8446c1efacf29525ff6713623029fae157be4`;
  vLLM-Ascend
  `df4b560fe8809dbb2b5099bc8e21ff4c25bfc466b8abf8f61a7ab188912e06d8`
- Executable hashes: SIMT v2
  `4c8f0d34b56153bd4816f4ca4c457598671c58ae25ff16fb609d5bf1623bb09e`;
  vLLM-Ascend
  `691dcb2275e23cd0f92934254b28a34cab505a9d7b0b0f63e78e5924a59b1553`

Both direct correctness runs passed. For every row the oracle found threshold
`112`, `2008` scores above threshold, and required/observed `40`
threshold-equal selections; bounds and uniqueness also passed.

| Implementation | Blocks | Five Task Duration samples (us) | Median (us) | Min (us) | Max (us) |
| --- | ---: | --- | ---: | ---: | ---: |
| vLLM-Ascend | 4 | 7.807, 7.946, 7.913, 7.709, 7.722 | 7.807 | 7.709 | 7.946 |
| SIMT v2 distributed | 64 | 27.792, 27.617001, 28.406, 27.916, 27.882 | 27.882 | 27.617001 | 28.406 |

The median ratio `vLLM-Ascend / SIMT v2` is `0.280001`; equivalently, the SIMT
v2 distributed stage takes `3.5714x` the vLLM-Ascend stage time for this
one-kernel boundary. The implementations deliberately retain their production
launch ownership, so their AIV block dimensions are not equal: forcing both to
the same block count would no longer measure either actual decode stage.

The supplied V3.2 workflow profile reports the SIMT production kernel
`lightning_indexer_decode_distributed_topk_bfloat16_v2_kernel` at `28.629` and
`28.742001 us`, both with block dimension 64. The standalone SIMT samples are
within about 1-4% of those fused-workflow rows, which is consistent with having
extracted the actual distributed decode path.

These measurements are device task time, not end-to-end host dispatch latency,
and do not generalize beyond the fixed BF16 decode shape and recorded Ascend
950/CANN environment.
