# Fused Timestamp LRU Metadata Update

## 1. Scope

This AIV-only operator is specialized for the sparse KV configuration used by
SGLang NPU DSA:

- `topk = 2048`
- `cache_capacity = 4096`
- all metadata uses contiguous `int32`
- valid request IDs start at row `0`
- one AIV owns one request row at a time (grid-stride when `B > blockDim`)
- A2/A3 scatter is not used

The operator returns `victim_slots[B, 2048]` and updates these tensors in place:

- `slot_map[R_map, W]`
- `device_lru_slots[R, 4096]`
- `device_lru_slot_stamps[R, 4096]`
- `device_slot_tokens[R, 4096]`

`device_lru_slots[row, i]` and `device_lru_slot_stamps[row, i]` are an aligned
pair. The persistent pair array is ordered by descending stamp: oldest first,
most recently hit/filled last.

## 2. Per-request algorithm

1. Load `lru_slots`, `lru_stamps`, and `device_token_pos` into UB.
2. Saturating SIMD increment:
   `stamp = min(stamp, stamp_max - 1) + 1`.
3. Create `C+K=6144` sort records:
   - base record: key `physical_slot + 0.25`, payload old LRU position;
   - hit record: key `device_token_pos`, payload `C + topk_position`;
   - miss/invalid hit records naturally use key `-1`.
4. Descending `Sort<float, true>` and `Extract`.
5. `CAST_RINT` maps both `slot+0.25` and `slot` to the physical slot. A base
   record is first in its group; equality with the next rounded slot means the
   slot was hit. Gather its incremented stamp and multiply by `!hit`.
6. Build a second-sort score for every record. Base records keep their updated
   non-negative stamp; auxiliary hit/miss records subtract `stamp_max + 1` and
   therefore become negative. The physical slot is carried as the sort index.
7. Run a second descending full sort over all 6144 records. The first
   4096 outputs are exactly aligned `(updated_stamp, physical_slot)` base pairs,
   already ordered by stamp. This avoids `GatherMask` compaction entirely.
8. Build the valid-miss vector with clamped SIMD arithmetic. Run an 11-round
   Hillis-Steele inclusive scan; each shift is an indexed `Gather`.
9. Gather `sorted_slots[miss_rank]` and restore `-1` for non-miss positions.
10. Load the complete `device_slot_tokens` row into UB. The scalar loop groups
    valid misses in batches of eight. It prepares aligned `new_token` and
    `victim` staging blocks with `SetValue`, synchronizes scalar-to-MTE3 once per
    batch, issues the 4-byte `DataCopyPad` writes below, then drains MTE3 once
    before reusing the staging blocks:
    - `slot_map[old_token] = -1` when the victim was occupied;
    - `device_slot_tokens[victim] = new_token`;
    - `slot_map[new_token] = victim`.
11. Reset the victim prefix stamps to zero, build the vector
    `(i + miss_count) % cache_capacity`, gather the rotated pairs into aligned
    full-row buffers, write the full LRU slot/stamp rows back to GM, and wait
    for MTE3 completion before the core reuses UB for another request.

Duplicate hits are safe because a slot has one base record followed by any
number of hit records. Top-k token IDs are expected to be unique for misses;
this is already guaranteed by the upstream top-k selector.

## 3. UB plan

The host reads the platform UB size and reserves 8 KiB for pipe overhead. The
fixed peak arena is 180,224 bytes (176 KiB):

| Region | Bytes | Lifetime |
|---|---:|---|
| 6144 keys + indices | 49,152 | first sort |
| first-sort temp + output pairs | 98,304 | first sort |
| old/compacted LRU pairs | 32,768 | first and second sort |

The 8 KiB device-token-position input reuses the idle first-sort temp region and
is consumed before `Sort` overwrites that region. Its first 544 bytes are later
reused for the immutable `-1` block and two eight-entry aligned scalar staging
areas used by batched sparse writes. After hit detection, the same
record-value/index, sort-temp, and sort-output regions are reused for the
second 6144-record sort, so pair compaction needs no additional UB. The miss
scan uses seven 8 KiB vectors, while sorted pairs and the delayed 16 KiB
slot-token row occupy non-overlapping regions. Including the pipe reserve, the
host-side UB requirement is 188,416 bytes.

## 4. Stream contract

`slot_map_lookup` runs on the caller stream. A `copy_ready` event releases one
copy stream, which runs the following kernels serially:

- D2D hit copy with 48 AIVs;
- H2D host-miss copy with 48 AIVs.

After both copies complete, the fused timestamp-LRU metadata update runs on its
own stream while the caller prepares sparse attention. Refill waits for metadata
completion, then uses `victim_slots` as destination indices. Invalid request
rows in `victim_slots` are left undefined and are ignored by the refill valid
mask.

## 5. Invariants and validation

- Request IDs in one launch are unique; valid IDs lie in `[0, R)`.
- `lru_slots` is a permutation of `[0, 4096)`.
- stamps lie in `[0, stamp_max]` and are non-increasing after writeback.
- `slot_map[token] == slot` iff `device_slot_tokens[slot] == token` for occupied
  slots.
- hit and invalid top-k positions of valid requests contain `-1`; output rows
  for invalid request IDs are undefined.
- `stamp_max <= 2^24-1`, keeping int32-to-float32 sort values exact.
