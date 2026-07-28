# msopprof Two-VF Standalone Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal standalone diagnostic that compares msopprof behavior for equivalent one-kernel workloads stored in device ELFs containing one versus two SIMT VF entries.

**Architecture:** Add two pure ACL/ASC executables under the Lightning Indexer operator-local test tree. The default paths both run one copy kernel; the two-VF executable also contains a second VF that can be launched with `--launch-second`, isolating ELF symbol layout from default launch count.

**Tech Stack:** CMake 3.16+, ASC language support, ACL runtime C API, Ascend SIMT API, `bisheng --npu-arch=dav-3510`, msopprof BasicInfo.

## Global Constraints

- Work only in `/root/aiagent/cannbench/.worktrees/lightning-indexer-context-split`.
- Keep the diagnostic under `src/cannbench/operators/builtin/lightning_indexer/simt/test/`.
- Do not depend on CannBench runtime code, Python, PyTorch, or torch_npu.
- Use only ACL and SIMT API in the `.asc` sources.
- The default workload for both executables must be one one-block copy launch.
- The control source must contain exactly one `__simt_vf__`; the reproduction source must contain exactly two.
- Do not modify Lightning Indexer production dispatch or decode behavior for this diagnostic.

---

### Task 1: Define And Verify The Standalone Source Contract

**Files:**
- Create: `src/cannbench/operators/builtin/lightning_indexer/simt/test/test_msopprof_two_vf_repro_source.py`

**Interfaces:**
- Consumes: files under `simt/test/msopprof_two_vf_repro/`.
- Produces: a test contract for CMake targets, VF counts, CLI switch, and dependency isolation.

- [ ] **Step 1: Add the failing source contract**

Add a test that reads `CMakeLists.txt`, `two_vf_repro.asc`, and
`single_vf_control.asc`, then asserts:

```python
assert "add_executable(two_vf_repro" in cmake
assert "add_executable(single_vf_control" in cmake
assert two_vf.count("__simt_vf__") == 2
assert single_vf.count("__simt_vf__") == 1
assert '"--launch-second"' in two_vf
for source in (two_vf, single_vf):
    assert '#include "acl/acl.h"' in source
    assert '#include "simt_api/asc_simt.h"' in source
    for forbidden in ("torch", "Python.h", "cannbench"):
        assert forbidden not in source
```

- [ ] **Step 2: Run the focused test and verify RED**

```bash
pytest -q \
  src/cannbench/operators/builtin/lightning_indexer/simt/test/test_msopprof_two_vf_repro_source.py
```

Expected: FAIL because the standalone directory does not exist.

### Task 2: Implement The Two Standalone Targets

**Files:**
- Create: `src/cannbench/operators/builtin/lightning_indexer/simt/test/msopprof_two_vf_repro/CMakeLists.txt`
- Create: `src/cannbench/operators/builtin/lightning_indexer/simt/test/msopprof_two_vf_repro/two_vf_repro.asc`
- Create: `src/cannbench/operators/builtin/lightning_indexer/simt/test/msopprof_two_vf_repro/single_vf_control.asc`
- Create: `src/cannbench/operators/builtin/lightning_indexer/simt/test/msopprof_two_vf_repro/README.md`

**Interfaces:**
- Consumes: device 0 and the Ascend runtime installed through `ASCEND_HOME_PATH`.
- Produces: executables `two_vf_repro` and `single_vf_control`, each returning zero only after direct output validation succeeds.

- [ ] **Step 1: Add the CMake build**

Use `find_package(ASC REQUIRED)`, `project(... LANGUAGES ASC CXX)`, and one
`add_executable` per `.asc` source. Set `LINKER_LANGUAGE ASC`, pass
`--npu-arch=${CMAKE_ASC_ARCHITECTURES}`, and include both standard Ascend
include locations for each target.

- [ ] **Step 2: Implement the single-VF control**

Define one `__simt_vf__` copy function, one `__global__ __vector__` wrapper,
and a direct ACL `main`. Allocate four `int32_t` elements, initialize them to
`{3, 5, 7, 11}`, launch one block with 32 SIMT threads, synchronize, copy the
result back, validate equality, clean up, and print:

```text
target=single_vf_control
sync_ret=0
validation=pass
```

- [ ] **Step 3: Implement the two-VF reproduction**

Define the same copy VF plus a second add-one VF and their separate vector
kernel wrappers. The default invocation launches only copy and validates
`{3, 5, 7, 11}`. With `--launch-second`, launch add-one after copy and validate
`{4, 6, 8, 12}`. Reject every other argument with exit code 2. Print the
target, `launch_second=true|false`, synchronization result, and validation.

- [ ] **Step 4: Document exact handoff commands**

The README must contain configure/build, direct runs, BasicInfo collection for
each target, expected affected/fixed profiler outcomes, and the source, build
log, direct-run log, msopprof log, and output directory list to send upstream.

- [ ] **Step 5: Run the source contract and verify GREEN**

```bash
pytest -q \
  src/cannbench/operators/builtin/lightning_indexer/simt/test/test_msopprof_two_vf_repro_source.py
git diff --check
```

Expected: both selected tests pass and no whitespace errors are reported.

### Task 3: Build, Run, And Profile On The Target Device

**Files:**
- Validate: `src/cannbench/operators/builtin/lightning_indexer/simt/test/msopprof_two_vf_repro/`

**Interfaces:**
- Consumes: the port-20002 device with CANN environment and the updated 100-second msopprof build.
- Produces: direct-run and separate BasicInfo profile artifacts for both targets.

- [ ] **Step 1: Sync to a fresh remote temporary directory**

Create `/tmp/cannbench-msopprof-two-vf-XXXXXX` using remote `mktemp -d` and
rsync the reproduction directory through SSH port 20002.

- [ ] **Step 2: Configure and build**

```bash
source /usr/local/Ascend/cann/set_env.sh
cmake -S . -B build -DCMAKE_ASC_ARCHITECTURES=dav-3510
cmake --build build --parallel
```

Expected: both executables build with exit code 0.

- [ ] **Step 3: Run all direct controls**

```bash
./build/single_vf_control
./build/two_vf_repro
./build/two_vf_repro --launch-second
```

Expected: every command exits zero and prints `validation=pass`.

- [ ] **Step 4: Profile equivalent default workloads separately**

```bash
msopprof --output="$PWD/profile-single" --aic-metrics=BasicInfo \
  --launch-count=1 ./build/single_vf_control
msopprof --output="$PWD/profile-two" --aic-metrics=BasicInfo \
  --launch-count=1 ./build/two_vf_repro
```

Record each exit code, profiler log, whether `RegisterFuncSymbol` appears, and
whether BasicInfo kernel data exists. Do not treat success on the rebuilt
profiler as failure of the sample; it means the minimal former trigger is fixed.

- [ ] **Step 5: Backfill observed results**

Add the remote directory, msopprof version, direct-run result, profile result,
and artifact locations to the reproduction README.

### Task 4: Verify And Commit The Diagnostic

**Files:**
- Modify: all files created or changed in Tasks 1-3.

**Interfaces:**
- Consumes: completed local and remote evidence.
- Produces: a self-contained diagnostic commit suitable for sharing upstream.

- [ ] **Step 1: Run operator-local and repository tests**

```bash
pytest -q src/cannbench/operators/builtin/lightning_indexer/simt/test
pytest -q
git diff --check
```

Expected: all suites pass and no whitespace errors are reported.

- [ ] **Step 2: Commit only the diagnostic and its source contract**

```bash
git add \
  src/cannbench/operators/builtin/lightning_indexer/simt/test/test_msopprof_two_vf_repro_source.py \
  src/cannbench/operators/builtin/lightning_indexer/simt/test/msopprof_two_vf_repro
git commit -m "test(lightning-indexer): add two-vf msopprof repro"
```

The existing 32-task fused source changes remain part of their separate
performance-gated implementation commit.
