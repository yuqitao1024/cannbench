# VCV per-AIV UB layout experiment

## Purpose

This standalone Ascend 950 example isolates the compiler behavior of static
`__ubuf__` arrays declared for different vector sub-blocks in one
`KERNEL_TYPE_MIX_AIC_1_2` translation unit. The experiment is intended for
binary and disassembly inspection, not as a general operator implementation or
a performance comparison.

## Fixed operation

- Launch one logical mixed AICore.
- Use BF16 `Q[8, 16]`, BF16 key table `[32, 16]`, and 16 fixed gather indices.
- V0 gathers 16 key rows into a statically declared 256-element BF16 UB array
  in one ZN fractal, then transfers it to L1 with `asc_copy_ub2l1_sync`.
- Cube duplicates the eight Q rows into both halves of a physical M=16 L1/L0A
  matrix, consumes the gathered L1 keys, and computes one fixed `16 x 16 x 16`
  BF16 MMAD with FP32 accumulation.
- Cube transfers the physical second M half directly from L0C into V1's UB
  with `asc_copy_l0c2ub_sync(..., sub_blockid=true, ...)`.
- V1 declares a 128-element FP32 logits array and a separate 256-element FP32
  scratch array, then computes `softmax[8,16]` and writes FP32 output to GM.

## Synchronization

The example intentionally uses the requested experimental exception:
`AscendC::CrossCoreSetFlag` and `AscendC::CrossCoreWaitFlag` with mode 4. V0
uses flag 0. V1 uses flag 17, where 16 selects the AIV1 intra-block flag bank.
All data movement and Cube compute use C API calls.

This exception is local to this compiler-observation example. It is not a
precedent for ordinary CannBench operator source, whose default boundary
forbids CrossCore Basic API synchronization.

## Correctness contract

The host builds the oracle from the exact BF16 input values, computes FP64 dot
products and softmax, and checks all 256 FP32 outputs with absolute tolerance
`2e-5` and relative tolerance `2e-4`. Every output row must also sum to one
within `1e-4`.

## Inspection contract

Source declarations alone do not prove effective UB allocation. Preserve the
build tree and inspect compiler metadata, device ELF files, and disassembly.
The relevant source anchors are `v0_gather_zn`, `v1_logits_row_major`, and
`v1_softmax_scratch`.
