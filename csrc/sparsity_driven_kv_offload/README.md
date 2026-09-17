# Sparsity-Driven KV Offloading Primitives

This directory contains the Ascend NPU kernel-layer primitives for the
[Sparsity-Driven KV Offloading RFC](https://github.com/sgl-project/sglang/issues/31779).
It does not include the SGLang runtime cache manager or sparse-attention backend
adapter.

## Modules

| Module | Purpose |
| --- | --- |
| `shm_allocator` | Allocates host-backed storage and registers it with the NPU, exposing a stable device-visible address. |
| `unidex_copy` | Performs masked indexed row copies for D2D, H2D, and D2H KV movement. |
| `slot_map_lookup` | Resolves sparse top-k logical KV positions against the device-resident slot map. |
| `fused_timestamp_lru_metadata_update` | Selects LRU victims and updates the slot metadata. |

The intended data path is:

```text
top-k indices
    │
    ▼
slot_map_lookup ── hits ───────────────┐
    │                                  │
    └── misses ── registered host KV ──┤
                                       ▼
                                  unidex_copy
                                       │
                                       ▼
                              selected top-k KV buffer
```

## Python API

The canonical Python API is:

```python
from sgl_kernel_npu.sparsity_driven_kv_offload import (
    create_shm_tensor,
    fused_timestamp_lru_metadata_update,
    free_shm,
    slot_map_lookup,
    unidex_copy_inplace,
)
```

The registered-memory lifecycle is process-local. The supported deployment
model is multiple processes with one NPU device bound to each process.

`slot_map_lookup(..., pos_mask_size=N)` additionally returns an int32 position
mask with shape `[bs, N]`. A cache hit at position `pos` sets
`position_mask[b, pos] = 1`; repeated hits remain binary, and hit positions
outside `[0, N)` are not written. `N` must be a multiple of 8 to support
aligned atomic mask updates. Omitting `pos_mask_size` preserves the legacy
two-output return value.

The fused LRU operator consumes this mask with `N=4096`. Reusing the lookup
result lets it preserve the already-sorted LRU order with one 4096-record
stable partition instead of matching hits and re-sorting 6144 records twice.
Its host wrapper then launches `parallel_lru_metadata_write` on the same stream.
That kernel divides each request into 64 tiles so miss-related slot-map and
reverse-map writes can use all available AIVs.

## Validation

Run the focused correctness and smoke benchmark suite:

```bash
scripts/sparsity_driven_kv_offload/test_unidex_shm.sh 0
```

Run benchmark sweeps:

```bash
scripts/sparsity_driven_kv_offload/sweep_unidex_copy.sh
scripts/sparsity_driven_kv_offload/sweep_slot_map_lookup.sh
```
