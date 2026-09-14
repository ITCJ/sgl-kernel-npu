# Fused Timestamp LRU Metadata Update

## 1. Scope

This AIV-only operator is specialized for the sparse KV configuration used by
SGLang NPU DSA:

- `topk = 2048`
- `cache_capacity = 4096`
- all metadata uses contiguous `int32`
- request row `0` is graph padding
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
6. `CompareScalar(payload < C)` builds a packed base-record mask. Two
   `GatherMask` calls compact exactly 4096 `(slot, updated_stamp)` pairs.
7. Sort the compacted pairs by stamp descending. Since every non-hit stamp was
   incremented to at least one and every hit stamp is zero, hits cannot be
   selected as victims even when all stamps started at zero.
8. Build the valid-miss vector with clamped SIMD arithmetic. Run an 11-round
   Hillis-Steele inclusive scan; each shift is an indexed `Gather`.
9. Gather `sorted_slots[miss_rank]` and restore `-1` for non-miss positions.
10. Load the complete `device_slot_tokens` row into UB. For each valid miss,
    the scalar loop reads addresses only and issues 4-byte `DataCopyPad` writes
    from 32-byte-aligned UB scalar staging blocks:
    - `slot_map[old_token] = -1` when the victim was occupied;
    - `device_slot_tokens[victim] = new_token`;
    - `slot_map[new_token] = victim`.
11. Reset the victim prefix stamps to zero, build the vector
    `(i + miss_count) % cache_capacity`, gather the rotated pairs into aligned
    full-row buffers, and write the full LRU slot/stamp rows back to GM.

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
is consumed before `Sort` overwrites that region. The packed base mask reuses
the record-value region after the rounded physical slots have been produced.
The stamp sort uses 96 KiB in the released first-sort area. The miss scan uses
seven 8 KiB vectors, while sorted pairs and the delayed 16 KiB slot-token row
occupy non-overlapping regions. Including the pipe reserve, the host-side UB
requirement is 188,416 bytes.

## 4. Stream contract

`slot_map_lookup` runs on the caller stream. A `copy_ready` event releases three
independent streams:

- D2D hit copy;
- H2D host-miss copy;
- fused timestamp-LRU metadata update.

The refill stream waits for both H2D miss completion and metadata completion,
then uses `victim_slots` as destination indices. Attention waits for refill and
metadata events before consuming the device cache state.

## 5. Invariants and validation

- Real request IDs in one launch are unique and lie in `[1, R)`.
- `lru_slots` is a permutation of `[0, 4096)`.
- stamps lie in `[0, stamp_max]` and are non-increasing after writeback.
- `slot_map[token] == slot` iff `device_slot_tokens[slot] == token` for occupied
  slots.
- hit and invalid output positions contain `-1`.
- `stamp_max <= 2^24-1`, keeping int32-to-float32 sort values exact.
