# Fused Timestamp LRU Metadata Update

## 1. Scope

This AIV-only operator is specialized for the sparse KV configuration used by
SGLang NPU DSA:

- `topk = 2048`
- `cache_capacity = 4096`
- all metadata uses contiguous `int32`
- valid request IDs start at row `0`
- one AIV owns one request row during victim selection (grid-stride when
  `B > blockDim`); metadata writes are tiled across AIVs
- A2/A3 scatter is not used

The selection operator returns `victim_slots[B, 2048]` and
`miss_counts[B]`, and updates these tensors in place:

- `device_lru_slots[R, 4096]`
- `device_lru_slot_stamps[R, 4096]`

The following `parallel_lru_metadata_write` operator consumes both outputs and
updates these tensors in place:

- `slot_map[R_map, W]`
- `device_slot_tokens[R, 4096]`

`device_lru_slots[row, i]` and `device_lru_slot_stamps[row, i]` are an aligned
pair. The persistent pair array is ordered by descending stamp: oldest first,
most recently hit/filled last.

## 2. Per-request algorithm

1. Load `lru_slots`, `lru_stamps`, and the 4096-entry physical-slot hit mask
   produced by `slot_map_lookup` into UB.
2. Saturating SIMD increment:
   `stamp = min(stamp, stamp_max - 1) + 1`.
3. Gather the hit bit by physical slot into the existing LRU order. Multiply
   incremented stamps by `!hit`, which resets hit timestamps to zero.
4. Stable-partition the existing LRU order with one 4096-record sort. Record
   `i` uses the unique integer key `!hit * 4096 + (4095 - i)` and carries `i`
   as payload. Descending sort places non-hits before hits while preserving the
   original order inside both groups. Since the persistent input is already in
   descending timestamp order, the result remains a valid LRU order.
5. Gather the reordered slots and stamps using the sorted original positions.
6. Build the valid-miss vector with clamped SIMD arithmetic. Run an 11-round
   Hillis-Steele inclusive scan; each shift is an indexed `Gather`.
7. Gather `sorted_slots[miss_rank]` and restore `-1` for non-miss positions.
8. Write one `miss_count` value per valid request for the following metadata
   kernel.
9. Reset the victim prefix stamps to zero, build the vector
   `(i + miss_count) % cache_capacity`, gather the rotated pairs into aligned
   full-row buffers, write the full LRU slot/stamp rows back to GM, and wait
   for MTE3 completion before the core reuses UB for another request.
10. Return `victim_slots` and `miss_counts` to the caller. The caller launches
    `parallel_lru_metadata_write` on the same stream. It splits every
    request into 64 independent 32-position tiles and distributes the tiles
    across all available AIVs. A tile copies its `topk_indices` and
    `victim_slots` into UB, loads aligned 32-byte reverse-map lines for its
    victims, and gathers all old tokens with SIMD. It then groups valid misses
    in batches of eight and issues these sparse writes:
    - `slot_map[old_token] = -1` when the victim was occupied;
    - `device_slot_tokens[victim] = new_token`;
    - `slot_map[new_token] = victim`.
    Requests with `miss_count == 0` skip their tiles before the UB copies.

Duplicate hits are safe because `slot_map_lookup` writes a binary mask with
atomic max. Top-k token IDs are expected to be unique for misses; this is
already guaranteed by the upstream top-k selector.

## 3. UB plan

The host reads the platform UB size and reserves 8 KiB for pipe overhead. The
fixed peak arena is 147,456 bytes (144 KiB):

| Region | Bytes | Lifetime |
|---|---:|---|
| 4096 keys + indices | 32,768 | stable-partition sort |
| sort temp + output pairs | 65,536 | stable-partition sort |
| old LRU pairs + hit mask | 49,152 | hit reset and reorder |

After `Extract`, the 32 KiB sort-output region holds the sorted slot/stamp
pairs. The physical-position-mask region is reused for the aligned
`miss_count` staging value. The miss scan uses seven 8 KiB vectors while the
sorted pairs and LRU writeback buffers occupy non-overlapping regions. The
fixed arena is 147,456 bytes; including the pipe reserve, the host-side UB
requirement is 155,648 bytes. The parallel metadata kernel uses about 2 KiB of
UB per AIV for two 32-entry input tiles, reverse-map lines, and batched scalar
staging.

## 4. Stream contract

`slot_map_lookup` runs on the caller stream. A `copy_ready` event releases one
copy stream, which runs the following kernels serially:

- D2D hit copy with 48 AIVs;
- H2D host-miss copy with 48 AIVs.

After both copies complete, the timestamp-LRU host operation launches victim
selection followed by the parallel metadata-write kernel on its own stream
while the caller prepares sparse attention. Same-stream ordering makes
`victim_slots` and `miss_count` visible to the second kernel without a host
synchronization. Refill waits for metadata completion, then uses
`victim_slots` as destination indices. Invalid request rows in `victim_slots`
are left undefined and are ignored by the refill valid mask.

## 5. Invariants and validation

- Request IDs in one launch are unique; valid IDs lie in `[0, R)`.
- `lru_slots` is a permutation of `[0, 4096)`.
- stamps lie in `[0, stamp_max]` and are non-increasing after writeback.
- `slot_map[token] == slot` iff `device_slot_tokens[slot] == token` for occupied
  slots.
- Victim slots and valid miss tokens are unique within one request, so metadata
  tiles write disjoint reverse-map and slot-map entries without atomics.
- hit and invalid top-k positions of valid requests contain `-1`; output rows
  for invalid request IDs are undefined.
- `stamp_max` fits positive int32. Sort keys are independent of timestamps and
  stay in `[0, 8191]`, so all keys are represented exactly by float32.
