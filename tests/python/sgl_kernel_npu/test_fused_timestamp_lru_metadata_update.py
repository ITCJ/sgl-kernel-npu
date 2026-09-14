import unittest

import sgl_kernel_npu  # noqa: F401
import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.sparsity_driven_kv_offload import (
    fused_timestamp_lru_metadata_update,
    slot_map_lookup,
)


class TestFusedTimestampLruMetadataUpdate(unittest.TestCase):
    TOPK = 2048
    CAPACITY = 4096
    MAX_CONTEXT_LEN = 8192

    def setUp(self):
        self.rows = 3
        self.slot_map = torch.full(
            (self.rows + 1, self.MAX_CONTEXT_LEN),
            -1,
            dtype=torch.int32,
            device="npu",
        )
        self.slot_tokens = torch.full(
            (self.rows, self.CAPACITY),
            -1,
            dtype=torch.int32,
            device="npu",
        )
        self.lru_slots = torch.arange(
            self.CAPACITY, dtype=torch.int32, device="npu"
        ).expand(self.rows, self.CAPACITY).clone()
        # Unique non-hit ages make victim order deterministic. Row zero is
        # padding and is checked for strict non-mutation.
        base_stamps = torch.arange(
            self.CAPACITY - 1, -1, -1, dtype=torch.int32, device="npu"
        )
        self.lru_stamps = base_stamps.expand(self.rows, self.CAPACITY).clone()

        self.slot_tokens[1, 0:3] = torch.tensor(
            [10, 20, 30], dtype=torch.int32, device="npu"
        )
        self.slot_map[1, 10] = 0
        self.slot_map[1, 20] = 1
        self.slot_map[1, 30] = 2

    def _run(self, row1_tokens, row2_tokens=(), stamp_max=(1 << 24) - 1):
        req_indices = torch.tensor([1, 2, 0], dtype=torch.int32, device="npu")
        topk = torch.full(
            (3, self.TOPK), -1, dtype=torch.int32, device="npu"
        )
        topk[0, : len(row1_tokens)] = torch.tensor(
            row1_tokens, dtype=torch.int32, device="npu"
        )
        if row2_tokens:
            topk[1, : len(row2_tokens)] = torch.tensor(
                row2_tokens, dtype=torch.int32, device="npu"
            )
        _, device_pos = slot_map_lookup(self.slot_map, req_indices, topk)
        padding_slots = self.lru_slots[0].clone()
        padding_stamps = self.lru_stamps[0].clone()
        padding_tokens = self.slot_tokens[0].clone()

        victims = fused_timestamp_lru_metadata_update(
            self.slot_map,
            req_indices,
            topk,
            device_pos,
            self.lru_slots,
            self.lru_stamps,
            self.slot_tokens,
            max_context_len=self.MAX_CONTEXT_LEN,
            stamp_max=stamp_max,
        )
        torch.npu.synchronize()
        return (
            victims.cpu(),
            padding_slots.cpu(),
            padding_stamps.cpu(),
            padding_tokens.cpu(),
        )

    def test_hit_reset_miss_evict_and_padding_skip(self):
        victims, padding_slots, padding_stamps, padding_tokens = self._run(
            [10, 40, 41], [50]
        )

        self.assertEqual(victims[0, :4].tolist(), [-1, 1, 2, -1])
        self.assertEqual(victims[1, :2].tolist(), [0, -1])
        self.assertTrue(torch.all(victims[2] == -1).item())

        self.assertEqual(self.slot_map[1, 10].item(), 0)
        self.assertEqual(self.slot_map[1, 20].item(), -1)
        self.assertEqual(self.slot_map[1, 30].item(), -1)
        self.assertEqual(self.slot_map[1, 40].item(), 1)
        self.assertEqual(self.slot_map[1, 41].item(), 2)
        self.assertEqual(self.slot_tokens[1, :3].cpu().tolist(), [10, 40, 41])

        slots = self.lru_slots[1].cpu()
        stamps = self.lru_stamps[1].cpu()
        self.assertTrue(torch.all(stamps[:-1] >= stamps[1:]).item())
        stamp_by_slot = dict(zip(slots.tolist(), stamps.tolist()))
        self.assertEqual(stamp_by_slot[0], 0)  # hit
        self.assertEqual(stamp_by_slot[1], 0)  # new fill
        self.assertEqual(stamp_by_slot[2], 0)  # new fill
        self.assertEqual(stamp_by_slot[3], self.CAPACITY - 3)

        self.assertTrue(torch.equal(self.lru_slots[0].cpu(), padding_slots))
        self.assertTrue(torch.equal(self.lru_stamps[0].cpu(), padding_stamps))
        self.assertTrue(torch.equal(self.slot_tokens[0].cpu(), padding_tokens))

    def test_duplicate_hits_reset_once_and_are_not_evicted(self):
        victims, *_ = self._run([10, 10, 40])
        self.assertEqual(victims[0, :4].tolist(), [-1, -1, 1, -1])
        slots = self.lru_slots[1].cpu()
        stamps = self.lru_stamps[1].cpu()
        stamp_by_slot = dict(zip(slots.tolist(), stamps.tolist()))
        self.assertEqual(stamp_by_slot[0], 0)
        self.assertEqual(stamp_by_slot[1], 0)

    def test_stamp_saturates(self):
        self.lru_stamps.fill_(7)
        victims, *_ = self._run([10], stamp_max=7)
        self.assertEqual(victims[0, 0].item(), -1)
        slots = self.lru_slots[1].cpu()
        stamps = self.lru_stamps[1].cpu()
        stamp_by_slot = dict(zip(slots.tolist(), stamps.tolist()))
        self.assertEqual(stamp_by_slot[0], 0)
        self.assertTrue(all(stamp_by_slot[i] == 7 for i in range(1, 32)))


if __name__ == "__main__":
    unittest.main()
