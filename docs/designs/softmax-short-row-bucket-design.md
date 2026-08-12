# Softmax V3 Short-Row Bucket Optimization

## Scope

This experiment used the existing published Ascend 950PR FP16 records to find
the largest remaining Softmax V3 gaps against CANN Ops. The leading groups
were width 49 (`1.964x`), width 144 (`1.703x`), width 256 (`1.521x`), width
196 (about `1.5x`), and 30K-50K logits (about `1.3x-1.5x`).

All implementation and dispatch changes remain in the Softmax operator
package. Dispatch depends only on dtype and reduction width; it does not use a
case id, model name, or fixed outer shape.

## Rejected 33-64 Candidate

The first candidate added a two-slot MTE2/V/MTE3 UB pipeline for widths 33-64
to target the largest published gap, width 49. FP16/FP32 correctness passed,
but repeated BasicInfo measurements showed that the fixed tile event and DMA
cost scaled poorly for large outer sizes:

| Case | Width | Baseline R1/R2 | Candidate R1/R2 | Result |
| --- | ---: | ---: | ---: | --- |
| `swin_window_attention` | 49 | 70.495 / 71.024 us | 81.119 / 80.932 us | 14-15% slower |
| `xcit_attention` | 48 | 4.474 / 4.524 us | 4.064 / 4.089 us | about 9% faster |

The candidate was rejected because a general shape bucket cannot trade a
large-outer regression for a small-outer gain. Its source and dispatch were
removed; widths below 129 retain the original generic path.

Remote evidence:

- baseline: `/root/cannbench-softmax-short-baseline-20260812`
- rejected candidate: `/root/cannbench-softmax-short-candidate-20260812`

## Retained 129-224 Buckets

The retained optimization specializes the existing 129-256 persistent UB
pipeline into compile-time capacity buckets:

```text
129 <= dim_size <= 160  -> kMaxElements = 160 (5 lane iterations)
161 <= dim_size <= 224  -> kMaxElements = 224 (7 lane iterations)
225 <= dim_size <= 256  -> kMaxElements = 256 (8 lane iterations, unchanged)
```

The optimization reduces inactive unrolled lane iterations and UB capacity
for an entire reduction-width class. It preserves the existing grid policy,
1024-thread VF, 32 rows per tile, DMA byte counts, event protocol, FP32
accumulation, and FP16/FP32 entry points.

The 160/224 instances live in `persistent_160_224.asc`. The original
`persistent_256.asc` remains byte-identical to the baseline and the build
appends the new object after all existing objects. This isolation was required
because a combined translation unit repeatedly changed width-256 performance:
candidate `119.342 us` versus baseline `118.013 us`. With isolation, the
width-256 control measured `117.928 us`, eliminating that regression.

## Ascend 950PR Evidence

Environment:

- node: port 20002, `Ascend950PR_9589`, `dav-3510`
- CANN: 9.2.0
- Bisheng: 15.0.5, `-O3`
- torch_npu: 2.10.0.post2
- profiler: `msopprof BasicInfo`, warmup 5, one kernel launch per record
- device frequency: 1650/1650 MHz for all accepted raw records

Final candidate root and provenance:

- root: `/root/cannbench-softmax-tight-isolated-20260812`
- `persistent_160_224.asc`: `13d18c7efd282daf04b0d5b98536d579618fe06cce415c82d1e9af4aa417b151`
- `row_persistent_fallback.asc`: `809faabc12dfbc3a5891537a5edeafe575961a523984021ec107aea4b0c495e0`
- clean build extension: `4c8b4878a2046181a0afeaecae10c52611fa7dc66ccb14b95a920b896843a254`
- inplace extension after final profiler instrumentation:
  `1a94b31efc8f73b99a4a8dbb77410a7c4329cd6702c9229b927bd2df7b0d8b6b`

Two repeated pre-isolation A/B rounds established the bucket effect. R2 is
shown below; the final isolated package was then measured again for publication.

| Case | Width | Baseline R2 | Tight bucket R2 | Gain |
| --- | ---: | ---: | ---: | ---: |
| `halonet_window_attention` | 144 | 232.441 us | 160.254 us | 31.06% |
| `levit_mixed_attention` | 196 | 351.881 us | 308.143 us | 12.43% |
| `levit_global_attention` | 196 | 700.816 us | 613.156 us | 12.51% |
| `crossvit_cls_attention` | 197 | 4.722 us | 4.320 us | 8.51% |
| `speech_transformer_attention` | 204 | 17.425 us | 15.738 us | 9.68% |

The final package passed FP16 and FP32 boundary comparisons at widths 129,
160, 161, 224, 225, and 256 with 65 tail rows. Maximum absolute error was at
most `9.54e-07` for FP16 and `1.49e-08` for FP32. The canonical FP16 suite
passed all 40 smoke, realistic, and stress cases.

The final publication collection from the isolated package was:

| Case | Final V3 | CANN Ops published | Final ratio |
| --- | ---: | ---: | ---: |
| `halonet_window_attention` | 159.441 us | 136.387 us | 1.169x |
| `levit_mixed_attention` | 305.089 us | 233.773 us | 1.305x |
| `levit_global_attention` | 606.981 us | 470.594 us | 1.290x |
| `crossvit_cls_attention` | 4.359 us | 4.396 us | 0.992x |
| `speech_transformer_attention` | 15.565 us | 14.793 us | 1.052x |
| `trocr_attention` (control) | 118.020 us | 77.596 us | 1.521x |

The clean build hash is the reproducible binary provenance. `msopprof`
rewrites the inplace copy while instrumenting kernels, so the post-profile
hash is recorded separately rather than presented as a build artifact.

Raw artifacts are under the candidate root in:

```text
correctness-buckets-isolated.log
accuracy-all-isolated-final.log
runs/softmax-tight-isolated-final-<case-id>/
```

## Publication Rule

Only the five records whose widths use the retained 160/224 buckets are
updated. Record order, schema, canonical run id, accuracy fields, and frontend
contract remain unchanged. The width-256 control and every other published
row are preserved.
