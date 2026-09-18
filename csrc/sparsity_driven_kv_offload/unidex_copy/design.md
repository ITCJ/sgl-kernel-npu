# `uindex_copy_optimized` blocked-interleaved scheduling design

## 1. Operator interface

```cpp
void uindex_copy_optimized(
    const at::Tensor &src,
    at::Tensor &dst,
    const at::Tensor &src_index,
    const at::Tensor &dst_index,
    const at::Tensor &valid_mask,
    int64_t src_rows,
    int64_t dst_rows,
    int64_t block_bytes,
    int64_t max_copy,
    int64_t block_dim = 48,
    c10::optional<int64_t> src_ptr = c10::nullopt,
    c10::optional<int64_t> dst_ptr = c10::nullopt);
```

For each `i < max_copy` whose mask and indices are valid, the operator copies
one logical byte row:

```text
dst[dst_index[i], 0:block_bytes] = src[src_index[i], 0:block_bytes]
```

`src` and `dst` may contain any PyTorch dtype because the kernel performs an
unmodified byte copy. `src_index` and `dst_index` are contiguous int64 NPU
tensors. `valid_mask` is a contiguous bool or uint8 NPU tensor. Optional raw
pointers support registered host memory.

## 2. Computation and implementation path

This is an indexed, memory-bound AscendC operator. No PyTorch primitive has
the same masked two-index in-place semantics, so it uses a custom AscendC
kernel.

The original kernel assigns one contiguous mapping interval to each core. For
a graph buffer padded to `max_running_requests`, a small active batch occupies
only the first few intervals. Assigning individual mappings cyclically gives
each core strided mask and index accesses. The optimized kernel instead assigns
contiguous 32-entry chunks cyclically:

```cpp
chunk_size = 32;
chunk_begin = GetBlockIdx() * chunk_size;
chunk_stride = GetBlockNum() * chunk_size;
for (; chunk_begin < max_copy; chunk_begin += chunk_stride) {
    chunk_end = min(chunk_begin + chunk_size, max_copy);
    for (i = chunk_begin; i < chunk_end; ++i) {
        if (!valid_mask[i]) continue;
        if (!indices_are_in_range(i)) continue;
        DataCopyPad(ub, src + src_index[i] * block_bytes, block_bytes);
        DataCopyPad(dst + dst_index[i] * block_bytes, ub, block_bytes);
    }
}
```

With `block_dim=48`, mappings 0-31 go to core 0, mappings 32-63 go to core 1,
and so on. After the first 1,536 mappings, core 0 receives the next 32-entry
chunk. Each mapping is inspected exactly once and each valid mapping still
uses one full-row GM-to-UB and UB-to-GM transfer.

## 3. Tiling strategy

### 3.1 Block-level tiling

No host tiling tensor is required. The launch dimension is `block_dim` and the
kernel derives its work directly from `GetBlockIdx()` and `GetBlockNum()`:

```text
chunk starts for core c = {(c + k * block_dim) * 32 | k >= 0}
```

Every core reads its mask in aligned 32-byte chunks and each int64 index array
in contiguous 256-byte chunks. For a valid prefix, useful rows are distributed
across all cores once the prefix reaches `block_dim * 32`; shorter prefixes use
`ceil(prefix_length / 32)` cores.

The original `unidex_copy` entry keeps contiguous partitioning as the
benchmark baseline. `uindex_copy_optimized` is a separate kernel entry in the
same source file.

### 3.2 UB-level tiling

One full logical row is one UB tile. A two-entry bound queue pipelines source
reads and destination writes.

| Buffer | Element type | Count | Bytes |
| --- | --- | ---: | ---: |
| `copyQue` | uint8 | 2 | `2 * align32(block_bytes)` |

The host limits `block_bytes` to 32 KiB, so queue usage is at most 64 KiB per
core. Byte copying does not perform arithmetic and does not require FP16 or
BF16 promotion.

## 4. Addressing and safety

The host validates that:

- tensor devices, dtypes, ranks, and contiguity match the interface;
- index and mask lengths are at least `max_copy`;
- row counts, byte sizes, launch dimension, and mapping count fit uint32;
- `rows * block_bytes` fits the kernel's uint32 byte-offset range;
- tensor storage is large enough when a raw registered-memory pointer is not used.

The kernel skips false masks, negative indices, and out-of-range indices.
Source and destination byte offsets are `row * block_bytes`.

## 5. Workspace and allocations

The operator requires no workspace and creates no temporary NPU tensors. Host
work is limited to validation, stream recording, address preparation, and one
kernel launch.

## 6. Performance plan

The design removes the former `max_copy * column_tiles` index expansion and
preserves large full-row DMA transfers. Total mapping checks are `max_copy`,
independent of `block_dim`.

Benchmark the complete operator for batch sizes 1, 4, and 16, multiple hit
rates, and D2D/H2D/D2H directions. Compare identical `block_dim` values against
the contiguous baseline. The main target is a padded decode batch where valid
entries occupy or are concentrated in the active prefix.

## 7. Implementation checklist

- [x] Add a separate `uindex_copy_optimized` AscendC kernel entry.
- [x] Use blocked cyclic mapping assignment and full-row `DataCopyPad` transfers.
- [x] Share host validation and launch preparation with `unidex_copy`.
- [x] Remove `column_tiles` and all mapping-tensor expansion.
- [x] Preserve raw registered-memory pointer support.
- [x] Cover padded prefixes, sparse masks, D2D, H2D, D2H, and idle-core cases.
- [ ] Compile with the target CANN environment.
- [ ] Run correctness tests and the focused benchmark sweep on NPU.
