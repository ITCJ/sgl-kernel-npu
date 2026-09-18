# `uindex_copy_optimized` host tiling design

## 1. Goal and interface

`uindex_copy_optimized` keeps the existing `unidex_copy` AscendC kernel and
changes only the mapping tensors and launch parameters prepared by the host.
It targets graph decode batches whose tensors reserve `MAX_RUNNING_REQUESTS`
rows while only a small prefix of requests is valid.

```text
uindex_copy_optimized(
    src, dst, src_index, dst_index, valid_mask,
    src_rows, dst_rows, block_bytes, max_copy,
    block_dim=48, column_tiles=0, src_ptr=None, dst_ptr=None) -> ()
```

`column_tiles=0` selects the largest divisor of `block_bytes` no greater than
`block_dim`. An explicit value must be positive, divide `block_bytes`, and not
exceed `block_dim`.

## 2. Host transformation

For `C=column_tiles`, each original byte row is viewed as `C` adjacent rows of
`tile_bytes=block_bytes/C`. The host expands the mapping tensors in column
major order:

```text
tiled_src_index[c, i] = src_index[i] * C + c
tiled_dst_index[c, i] = dst_index[i] * C + c
tiled_valid_mask[c, i] = valid_mask[i]
```

It then launches the existing kernel with:

```text
src_rows'   = src_rows * C
dst_rows'   = dst_rows * C
block_bytes'= block_bytes / C
max_copy'   = max_copy * C
block_dim'  = C
```

Because `max_copy'` is exactly divisible by `C`, core `c` receives one full
column containing all original mapping entries. Valid requests therefore
appear on every launched core even when valid rows occupy only a small prefix
of the graph buffer.

For a 1152-byte KV row, auto tiling selects 24 columns of 48 bytes for each of
the two overlapping hit/miss copies, and 48 columns of 24 bytes for refill.

## 3. Correctness

For every valid original mapping `i`, the `C` transformed byte intervals are
disjoint and their ordered union is the original interval:

```text
union(c=0..C-1) [row * block_bytes + c * tile_bytes,
                 row * block_bytes + (c + 1) * tile_bytes)
```

The source and destination use the same `C`, so the transformed copies retain
the original byte ordering. Invalid mappings repeat a false mask and perform
no writes. Raw registered-memory pointers remain base addresses; transformed
row offsets are relative to those same addresses.

## 4. Kernel and UB usage

`op_kernel/unidex_copy_kernel.cpp` is unchanged. Each core still uses the
existing two-entry `TQueBind` and `DataCopyPad` pipeline. Per-core UB usage is
reduced from `2 * align32(block_bytes)` to
`2 * align32(block_bytes / C)`. No workspace is required.

## 5. Cost and benchmark

The host creates two int64 index tensors and one mask tensor with
`max_copy * C` entries. This trades index-generation bandwidth for better copy
parallelism, so the benchmark reports the complete operator latency, including
host-side NPU tensor expansion and the reused copy kernel. Compare it with
`unidex_copy` over actual batch sizes 1, 4, and 16 before choosing a production
threshold for a specific model and device.

