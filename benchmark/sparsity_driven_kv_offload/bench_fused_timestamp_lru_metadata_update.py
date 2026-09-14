import argparse
import time

import sgl_kernel_npu  # noqa: F401
import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.sparsity_driven_kv_offload import (
    fused_timestamp_lru_metadata_update,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    batch = args.batch_size
    rows = batch + 1
    topk = 2048
    capacity = 4096
    max_context = 8192
    req = torch.arange(1, rows, dtype=torch.int32, device="npu")
    tokens = torch.arange(topk, dtype=torch.int32, device="npu").expand(
        batch, topk
    ).contiguous()
    device_pos = torch.full_like(tokens, -1)
    slot_map = torch.full(
        (rows + 1, max_context), -1, dtype=torch.int32, device="npu"
    )
    lru_slots = torch.arange(
        capacity, dtype=torch.int32, device="npu"
    ).expand(rows, capacity).clone()
    lru_stamps = torch.zeros_like(lru_slots)
    slot_tokens = torch.full_like(lru_slots, -1)

    def run_once():
        fused_timestamp_lru_metadata_update(
            slot_map,
            req,
            tokens,
            device_pos,
            lru_slots,
            lru_stamps,
            slot_tokens,
            max_context_len=max_context,
        )

    for _ in range(args.warmup):
        run_once()
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(args.iters):
        run_once()
    torch.npu.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000 / args.iters
    print(f"batch={batch} mean_latency_ms={elapsed_ms:.4f}")


if __name__ == "__main__":
    main()
