# sparse_attention SIMT

This directory contains Ascend SIMT integration for `sparse_attention`.

Fast-path families:

- `family_hd512`
- `family_hd128`

Unsupported shapes are rejected by the SIMT plugin path.
