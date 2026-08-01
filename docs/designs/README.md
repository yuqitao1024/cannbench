# Design Archive

This directory contains durable architecture and operator design records.
Implementation plans are intentionally not archived here: current work should
be tracked in code, tests, issues, or an active external task system.

Design filenames describe the subject without a creation-date prefix. Dates
inside a document remain historical metadata.

## Superpowers Archive Audit

The following documents were previously generated as Superpowers specs and
checked against the current codebase on 2026-07-30.

| Design | Status | Current outcome |
| --- | --- | --- |
| [CUDA treasure map](cuda-treasure-map-design.md) | Not implemented | No modal, route component, or dark-theme trigger exists. |
| [DSA API boundary contraction](dsa-api-boundary-contraction-design.md) | Not completed | Target score kernels still use disallowed Basic API facilities. |
| [Lightning Indexer context-sharded decode](lightning-indexer-context-sharded-decode-design.md) | Superseded | The two-launch path was implemented, then replaced for enabled tiers by the parameterized single kernel. |
| [Lightning Indexer 32 mixed-task prefill](lightning-indexer-32-mixed-task-prefill-design.md) | Implemented | The retained common path uses up to 32 mixed tasks and 1024-thread VFs. |
| [Lightning Indexer decode single kernel](lightning-indexer-decode-single-kernel-design.md) | Implemented | S16, S8, S4, S2, and S1 tiers are enabled after validation. |
| [Lightning Indexer V3.2 prefill row parallelism](lightning-indexer-v32-prefill-row-parallel-design.md) | Closed | The Q=2 candidate was 11.11% slower and remains disabled; the common-path correction was retained. |
| [msopprof two-VF reproduction](msopprof-two-vf-repro-design.md) | Partially resolved | The sample is complete, but the historical failure needs unavailable affected binaries to reproduce. |
| [Sparse Attention V3.2 prefill Head64](sparse-attention-v32-prefill-head64-design.md) | Implemented | The exact case uses the 32-AIC/64-AIV Head64 P1 direct-output path. |

## Other Designs

- [Ascend SIMT DSA operators](ascend-simt-dsa-operators-design.md)
- [DSA inference fusion](dsa-inference-fusion-spec.md)
- [DSA HD256/HD576 CV fusion](dsa-hd256-hd576-cv-fusion-design.md)
- [DSA V2 decode profile-guided optimization](dsa-v2-decode-profile-guided-optimization-design.md)
- [DSA V3.2 performance timing boundary](dsa-v32-performance-timing-boundary-design.md)
- [DSA V3.2 three-path semantics](dsa-v32-three-path-semantics.md)
- [DSA V4/V4Pro alignment backlog](dsa-v4-v4pro-alignment-backlog.md)
- [Lightning Indexer FP16 fused kernel](lightning-indexer-fp16-fused-kernel-design.md)
- [Lightning Indexer SIMT custom-op V1](lightning-indexer-simt-custom-op-v1-design.md)
- [Lightning Indexer V2 unordered radix Top-K](lightning-indexer-v2-unordered-radix-topk-design.md)
- [Operator and hardware visualization](operator-and-hardware-visualization-design.md)
- [Sparse Attention HD128 prefill fusion](sparse-attention-hd128-prefill-fused-design.md)
