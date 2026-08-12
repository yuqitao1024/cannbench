# VCV per-AIV UB layout

This is a fixed-shape Ascend 950 compiler experiment for a V0 -> Cube -> V1
pipeline. It answers a narrow question: when V0 and V1 declare different
static `__ubuf__` arrays in one mixed-kernel source, does the compiler account
for them independently on the two physical vector units, or make both layouts
effective on each vector unit?

The data path is explicit:

```text
V0: GM key table -> gather in v0_gather_zn -> UB2L1
Cube: duplicate GM Q[8,16] into physical M=16 -> MMAD -> L0C2UB(sub-block 1)
V1: v1_logits_row_major -> v1_softmax_scratch -> GM output
```

The physical Cube M is 16 because a single-destination `L0C2UB` targeting V1
selects the second M half on the tested `dav-3510` toolchain. The same eight Q
rows are placed in both halves, so V1 receives one complete logical
`softmax[8,16]` input without writing the result into V0's UB.

The two boundaries use `CrossCoreSetFlag/CrossCoreWaitFlag` mode 4. This is an
experimental, user-requested exception to the repository's normal operator API
boundary and is confined to this standalone example.

## Build and run

On an Ascend 950 node with CANN 9.2.0:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash scripts/run.sh
```

The runner deliberately retains `build/`. A successful run ends with
`Verification PASSED`.

## Inspecting UB allocation

Do not infer physical UB consumption from the three source arrays alone. Use
the retained compiler outputs to correlate the mixed kernel, its two vector
sub-block programs, resource metadata, and disassembly. Useful starting points:

```bash
find build -type f | sort
find build -type f \( -name '*.o' -o -name '*.so' -o -name '*.json' -o -name '*.bin' \) -print
```

The intended contrast is:

- V0 static UB: `v0_gather_zn`, 512 bytes.
- V1 static UB: `v1_logits_row_major` plus `v1_softmax_scratch`, 1536 bytes.

The key expected structural evidence is that V0 does not acquire V1's 1536
bytes and V1 does not acquire V0's 512 bytes merely because both branches were
compiled from the same source.
