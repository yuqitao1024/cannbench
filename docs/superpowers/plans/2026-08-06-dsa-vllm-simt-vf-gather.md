# DSA vLLM SIMT VF Gather Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an operator-local `vllm` SIMT version of V3.2 decode sparse attention, reproduce the copied vLLM-Ascend implementation, and measure separate QK-gather and V-gather SIMT VF conversions in the CannBench decode workflow.

**Architecture:** The new implementation lives entirely under `src/cannbench/operators/builtin/sparse_attention/simt/vllm/`. The sparse-attention plugin maps `--implementation-version vllm` to an isolated Python module and prepares the existing V3.2 TND contract outside the measured callable. The copied vLLM-Ascend MLA source remains attributable and independently editable; the existing `dsa_decode` plugin continues to own workflow expansion and dataset mapping.

**Tech Stack:** Python 3.11, PyTorch 2.9, torch-npu 2.9, CANN 9.2/dav-3510, Bisheng/AscendC Basic API plus SIMT VF, pytest, CannBench remote runner and profiler.

## Global Constraints

- Base the worktree on local `main` commit `c4db9d7`.
- Preserve the BF16 V3.2 decode contract: `B=2`, `Q=2`, `Hq=128`, `Hkv=1`, `context=32768`, `selected=2048`, `Dqk=576`, `Dv=512`, causal output plus LSE.
- This explicitly function-first version may retain copied vLLM-Ascend Basic API code; new gather experiments may use SIMT VF without first contracting to the repository's target API boundary.
- Keep all concrete DSA workflow and implementation rules in operator packages; do not edit public backend, CLI, config, or result layers.
- Compare clean-process runs under one remote environment, with identical inputs, warmup, repetitions, device, frequency, and workflow boundary.
- Retain source/package hashes, loaded paths, raw correctness output, raw timings, kernel launches, and profiler artifacts for every retained comparison.

---

### Task 1: Register the isolated SIMT version

**Files:**
- Modify: `src/cannbench/operators/builtin/sparse_attention/__init__.py`
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_dispatch.py`
- Create: `src/cannbench/operators/builtin/sparse_attention/simt/vllm/aten_dsa_sparse_attention_vllm/__init__.py`
- Create: `src/cannbench/operators/builtin/sparse_attention/simt/vllm/aten_dsa_sparse_attention_vllm/ops.py`

- [ ] Add failing assertions that version `vllm` resolves to `aten_dsa_sparse_attention_vllm` and has its own installable project.
- [ ] Run the focused test and confirm it fails because the version is not registered.
- [ ] Add the operator-local module mapping and a wrapper with the existing `sparse_attention_forward` result contract.
- [ ] Re-run the focused test and the sparse-attention plugin tests.

### Task 2: Import the deployed vLLM-Ascend MLA source with provenance

**Files:**
- Create: `src/cannbench/operators/builtin/sparse_attention/simt/vllm/vendor/sparse_flash_attention/*.h`
- Create: `src/cannbench/operators/builtin/sparse_attention/simt/vllm/vendor/sparse_flash_attention/sparse_flash_attention.cpp`
- Create: `src/cannbench/operators/builtin/sparse_attention/simt/vllm/vendor/PROVENANCE.md`
- Create: `src/cannbench/operators/builtin/sparse_attention/simt/vllm/setup.py`
- Create: `src/cannbench/operators/builtin/sparse_attention/simt/vllm/install.sh`

- [ ] Add a source-layout test requiring the entry, common, kernel, cube-service, vector-service, and tiling-key files.
- [ ] Run it red before copying.
- [ ] Copy those files byte-for-byte from the remote vLLM-Ascend 0.18.0 installation and record SHA-256 hashes and license origin.
- [ ] Add the minimal local build/install packaging without changing copied source.
- [ ] Verify local source hashes match the remote installed source hashes.

### Task 3: Reproduce the copied V3.2 decode path

**Files:**
- Modify: `src/cannbench/operators/builtin/sparse_attention/__init__.py`
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/vllm/aten_dsa_sparse_attention_vllm/ops.py`
- Create: `src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_vllm_source.py`

- [ ] Add failing tests for V3.2-only dispatch, TND query/key/key-rope/value/index shapes, output/LSE normalization, and rejection of unsupported shapes.
- [ ] Reuse operator-local lowering helpers so setup/materialization stays outside the timed closure.
- [ ] Build and deploy to a unique remote directory; prove the loaded module/source and binary hashes.
- [ ] Run direct real-device accuracy against the existing torch reference, including output and LSE.
- [ ] Run two clean-process CannBench `dsa_decode/realistic` workflow measurements and one BasicInfo profile; retain raw outputs.

### Task 4: Convert QK gather to SIMT VF

**Files:**
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/vllm/vendor/sparse_flash_attention/sparse_flash_attention_service_vector_mla.h`
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_vllm_source.py`

- [ ] Add a failing source/ABI test that identifies the QK gather boundary and requires the dedicated SIMT VF implementation.
- [ ] Replace only QK key/key-rope gather, preserving its workspace layout, row ownership, tail handling, and synchronization.
- [ ] Rebuild in a fresh remote package/process, then run the same output/LSE accuracy cases.
- [ ] Run the same two clean-process workflow measurements and BasicInfo profile; compare distribution and launch parity to copied baseline.

### Task 5: Convert V gather to SIMT VF

**Files:**
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/vllm/vendor/sparse_flash_attention/sparse_flash_attention_service_vector_mla.h`
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_vllm_source.py`

- [ ] Add a failing test for the V gather VF boundary without weakening the QK assertion.
- [ ] Replace only V gather, preserving probability/PV consumption layout, selected-token tails, and rolling-slot synchronization.
- [ ] Rebuild in a fresh remote package/process and repeat exact output/LSE accuracy.
- [ ] Repeat the identical workflow measurements and BasicInfo profile; report copied, QK-VF, and QK+V-VF results side by side.

### Task 6: Final verification and evidence handoff

**Files:**
- Create: `docs/optimization/dsa-vllm-simt-vf-gather-results.zh-CN.md`

- [ ] Run focused operator tests, architecture hardcoding searches, and full `pytest -q`.
- [ ] Verify the final remote source/package hash and loaded path in the benchmark process.
- [ ] Document device/toolchain versions, exact commands, cases, tolerance, warmup/repetitions, raw artifact paths, launches, medians/spread, regressions, and unsupported paths.
- [ ] Review `git diff --check`, worktree status, and the requirement checklist before reporting completion.
