# DSA Decode TopK SIMT VF vs SIMD Micro API

This fixed-shape example compares the retained five-stage SIMT v2 distributed
radix TopK with a same-dataflow SIMD Micro API rewrite. Both executables consume
host-generated BF16 scores shaped `[4,32768]`, return INT32 indices shaped
`[4,2048]`, launch one 64-block kernel, and use the same 45,056-byte dynamic UB
and GM workspaces.

See `SPEC.md` for the frozen semantic and measurement contract.

## Run

On the Ascend 950PR node with the CANN environment loaded:

```bash
RESULT_ROOT=/tmp/dsa-topk-simd-micro-exp-unique ./scripts/run.sh
```

The runner builds both binaries, requires direct correctness, then collects ten
alternating-order `Default` profile pairs. It rejects any parsed row that is not
one exact 64-block target launch at `1650/1650 MHz`.

After `summary.json` is written, it creates the deterministic sibling evidence
files `${RESULT_ROOT}.tar.gz` and `${RESULT_ROOT}.tar.gz.sha256`. The checksum
contains the SHA-256 for the archive, so a controller can copy both files with
`scp` and verify the archive independently of the remote result directory.

## Result

### Recorded fixed-shape result

The ten accepted alternating-order pairs show that the same-dataflow SIMD Micro
rewrite regressed. The retained SIMT v2 baseline median was `14.191500 us` and
the SIMD Micro candidate median was `25.939000 us`: a candidate improvement of
`-82.778424%`, equivalently an `82.778424%` median regression. The candidate
lost all ten pairs, so it is experimental comparison code only and is not
retained as an optimization.

| Round | SIMT v2 baseline (us) | SIMD Micro (us) | Candidate - baseline (us) |
| ---: | ---: | ---: | ---: |
| 1 | 14.281000 | 25.899000 | +11.618000 |
| 2 | 14.284000 | 25.767000 | +11.483000 |
| 3 | 14.227000 | 26.118000 | +11.891000 |
| 4 | 14.352000 | 25.684000 | +11.332000 |
| 5 | 14.249000 | 26.063999 | +11.814999 |
| 6 | 14.127000 | 25.983999 | +11.856999 |
| 7 | 14.046000 | 25.927000 | +11.881000 |
| 8 | 14.148000 | 25.951000 | +11.803000 |
| 9 | 14.156000 | 25.750999 | +11.594999 |
| 10 | 14.091000 | 26.238001 | +12.147001 |

| Statistic | SIMT v2 baseline | SIMD Micro candidate |
| --- | ---: | ---: |
| Median | 14.191500 us | 25.939000 us |
| Minimum | 14.046000 us | 25.684000 us |
| Maximum | 14.352000 us | 26.238001 us |

The paired candidate-minus-baseline median is `+11.8089995 us`; candidate
wins/ties/baseline wins are `0/0/10`. Every archived parsed row names its
expected target kernel, has exactly one target launch, uses block dimension
`64`, and reports current/rated execution frequency `1650/1650 MHz`.

### Measurement and environment

- Remote endpoint: `root@121.41.199.170:20002`; captured host: `tools2` at
  `2026-08-18T07:18:29+08:00`.
- Target: Ascend 950PR, `dav-3510`; CANN root:
  `/usr/local/Ascend/cann-9.2.0`; driver: `7.0.t9.0.B791`.
- Toolchain: Bisheng clang `15.0.5` (`clang-5c68a1cb1231`); profiler:
  `msopprof 26.2.0-9f53f86c867875b638157649a76e7d8ce8d32636`.
- Both direct executable logs report `Verification PASSED`. Each pair was
  profiled in alternating order (baseline first in odd rounds) with:

  ```bash
  msopprof --output=<unique-raw-directory> --aic-metrics=Default --launch-count=1 <executable>
  ```

  The command has no profiler warmup setting and no `--kernel-name` filter.
  The parser accepted only one exact target-kernel row per profile, with a
  64-block launch and `1650/1650 MHz`; no rejected or aggregate rows enter the
  table.

### Evidence identity and locations

The captured manifest reports `source_revision=unavailable`; the following
hashes therefore identify the measured source and executables directly:

| Artifact | SHA-256 |
| --- | --- |
| `simt_v2_topk.asc` | `d0f5821eeb739d00c2ab783c0bc19a401a1500e3b5918746b0c8d07ded13a0ae` |
| `simd_micro_topk.asc` | `a559510c09b204b60ccb110dcc7d398590a77abf016f1bdb939a8dc7ee6a9c92` |
| `host_common.h` | `47418b10769d9faceb029e4f8aa869bac063b6cf1d9f23b87c34c338a4913d64` |
| `dsa_decode_topk_simt_v2_baseline` | `3f5c8a011861fdcb21c4b1ea3c5245a8d46ecf51ff15b4806d0b6a8f5f2da233` |
| `dsa_decode_topk_simd_micro` | `a0fb8ecca672fdc06429eb2634cf31f478dc6647bded64fe7a16ca49fcb6c4d4` |
| Evidence archive | `f2ac651ba456648706e5f98dce115681fd865f194a557e0c767f60d01f666cee` |

- Remote evidence root:
  `/tmp/dsa-topk-simd-micro-final-run-zuxHN2/evidence`
- Remote archive:
  `/tmp/dsa-topk-simd-micro-final-run-zuxHN2/evidence.tar.gz`
- Local verified archive:
  `/tmp/dsa-topk-simd-micro-final-run-zuxHN2-evidence.tar.gz`

The local archive SHA-256 was independently checked against the recorded
archive hash. It contains `summary.json`, environment and device manifests,
source/executable hashes, direct correctness logs, exact profile commands,
all 20 parsed rows, and raw OPPROF directories.
