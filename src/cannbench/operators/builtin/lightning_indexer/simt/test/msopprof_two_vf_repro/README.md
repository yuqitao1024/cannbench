# msopprof two-VF standalone reproduction

This sample compares two equivalent pure-Vector workloads:

- `single_vf_control` links one 1024-thread SIMT VF and launches one copy
  kernel.
- `two_vf_repro` links two 1024-thread SIMT VFs and launches the same copy
  kernel.

Both kernel entries use `__global__ __vector__`; there is no AIC path or mixed
CV task declaration. The optional `--launch-second` argument makes
`two_vf_repro` launch its add-one VF after the copy. Both kernels read the
immutable input buffer so profiler warm-up replays do not alter validation.
The sample uses ACL and ASC directly; it does not use Python, PyTorch,
torch_npu, or CannBench runtime code.

## Build

```bash
source /usr/local/Ascend/cann/set_env.sh
cmake -S . -B build -DCMAKE_ASC_ARCHITECTURES=dav-3510
cmake --build build --parallel 2 2>&1 | tee build.log
```

## Direct validation

Run these before profiling:

```bash
./build/single_vf_control | tee direct-single.log
./build/two_vf_repro | tee direct-two.log
./build/two_vf_repro --launch-second | tee direct-two-both.log
```

Every command must exit zero and print `validation=pass`.

## Profile

Use separate output roots so the two results cannot overwrite each other:

```bash
PROFILE_ROOT=$(mktemp -d /tmp/msopprof-two-vf-XXXXXX)
msopprof \
  --output="$PROFILE_ROOT/single" \
  --aic-metrics=BasicInfo \
  --launch-count=1 \
  ./build/single_vf_control 2>&1 | tee msopprof-single.log

msopprof \
  --output="$PROFILE_ROOT/two" \
  --aic-metrics=BasicInfo \
  --launch-count=1 \
  ./build/two_vf_repro 2>&1 | tee msopprof-two.log
```

On an affected profiler build, the single-VF control should produce BasicInfo
data while the two-VF executable may fail or produce no kernel data. The
historical operator failure was `507015: VEC VF instruction parameter invalid`;
do not conflate it with the later non-fatal `RegisterFuncSymbol` warnings seen
from Sparse Attention. On a fixed build, both profiles should succeed.

## Handoff bundle

Send these files to the profiler maintainer:

- this source directory;
- `build.log`;
- `direct-single.log`, `direct-two.log`, and `direct-two-both.log`;
- `msopprof-single.log` and `msopprof-two.log`;
- the two directories below `$PROFILE_ROOT`;
- the output of `msopprof --version`.

## Historical incident evidence

The original July 26 session was searched again using the timestamp of commit
`673eda8 fix(dsa): isolate SIMT VF device libraries` as the anchor. The
following evidence comes from the recorded command output, not from published
benchmark summaries. Times below are China Standard Time (UTC+8):

- At 20:00, the original linked ELF placed the 4x64
  `vector_simt_entry` at `.text+0x0f30` and the 64x128 entry at
  `.text+0x18a0`. The first 4x64 VF profiled successfully; the second 64x128
  VF failed.
- At 20:04, swapping the ASC/link order moved the 64x128 VF to the first
  position. It then profiled successfully with a `164.811 us` task duration.
- At 20:05, the reverse check launched 4x64 after the swap. The now-second
  4x64 VF failed with the same error:

  ```text
  507015: VEC VF instruction parameter invalid
  ```

- At 21:53, the operator-side isolation fix was committed as `673eda8`.
- At 23:07, the same target session recorded the profiler combination as:

  ```text
  msopprof version:    26.1.0-05185f7d50b2abcabb2132dbe01c1ef3a4629aa0
  msopscommon version: 1026376e17f8ef704f2ef31fa81455c1cbc62726
  ```

This was a bidirectional order reproduction: whichever VF occupied the second
`vector_simt_entry` position failed, regardless of whether it was the 4x64 or
64x128 implementation. The evidence therefore ties the historical failure to
the second VF position in the monolithic device ELF, rather than to either
family's computation. The exact affected binaries were not retained, so this
version identification is historical session evidence rather than a currently
rerunnable artifact.

## Latest target-device result

Validated on the port-20002 target on July 28, 2026:

```text
pure-Vector source/build/log root: /tmp/cannbench-msopprof-pure-vector-mQL5kV
installed msopprof:                26.0.0-4c8a6f0800099b860c0d6c8506b55236fe27b39d
installed msopscommon:             20eab3a33253aa02e1b97c500e4070515aa9d640
26.1 branch test runtime:           /tmp/msopprof-26.1-vector-DctM4e/runtime-b4ff
26.1 branch msopprof commit:        5e79874a758104fcfa147b47c59e6adaa812ead7
26.1 branch msopscommon commit:     b4ffdb3d6ba7acdb7bbfcf359f6e4dd5b587a4f4
```

Both targets built successfully and all direct runs passed. With the installed
profiler, the one-VF and two-VF default profiles produced BasicInfo data in
`1.931 us` and `2.291 us`, respectively. The replay-safe `--launch-second`
profile also passed and produced separate BasicInfo CSV files for copy
(`1.760 us`) and add-one (`1.790 us`). No run reported
`RegisterFuncSymbol`.

The independently built 26.1 branch combination also profiled the pure-Vector
one-VF and two-VF controls successfully in `2.366 us` and `2.410 us`. This is
the latest public 26.1 branch combination, not the unavailable historical
`05185f7d50b2abcabb2132dbe01c1ef3a4629aa0` msopprof plus
`1026376e17f8ef704f2ef31fa81455c1cbc62726` msopscommon combination.

A matching mixed-CV comparison under
`/tmp/cannbench-msopprof-shared-mixed-qFYASb` also succeeded for one and two VF
entries with both profiler combinations. Mixed-CV replay emitted probe-symbol
and link-ordering warnings, but still produced valid BasicInfo data. An older
`c6689cc18cfcb75623b4cf7d3622c89274095170` profiler failed both the one-VF and
two-VF pure-Vector controls with `Kernel binary register failure`, so that
failure is not specific to multiple VF entries.

The controlled results show that neither two VF entries nor mixed-CV layout is
a sufficient trigger on the available compatible profiler builds. Reproducing
the historical failure still requires the exact affected profiler binary or
another property of the original operator ELF.
