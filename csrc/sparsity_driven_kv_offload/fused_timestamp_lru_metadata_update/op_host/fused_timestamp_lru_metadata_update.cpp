// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
// you may not use this file except in compliance with the License.

#include "defines.h"
#include "torch_helper.h"
#include "tiling/platform/platform_ascendc.h"

#include "aclrtlaunch_fused_timestamp_lru_metadata_update.h"

#include <algorithm>
#include <limits>

namespace sglang {
namespace npu_kernel {

namespace {

constexpr uint32_t kFixedTopk = 2048;
constexpr uint32_t kFixedCacheCapacity = 4096;
constexpr uint32_t kAlignment = 8;
constexpr uint32_t kPipeReserveBytes = 8 * 1024;
constexpr uint32_t kRequiredWorkUbBytes = 180224;
constexpr int64_t kMaxExactFp32Stamp = 16777215;
constexpr uint64_t kUint32Max = std::numeric_limits<uint32_t>::max();

void CheckNpuTensor(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.device().type() == at::DeviceType::PrivateUse1, name, " must be on an NPU device");
}

void CheckSameDevice(const at::Tensor &tensor, const at::Tensor &reference, const char *name)
{
    TORCH_CHECK(tensor.device() == reference.device(), name, " must be on the same device as slot_map");
}

void CheckInt32Contiguous(const at::Tensor &tensor, const char *name)
{
    CheckNpuTensor(tensor, name);
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == at::kInt, name, " must be int32, got ", tensor.scalar_type());
}

void CheckFitsUint32(int64_t value, const char *name)
{
    TORCH_CHECK(value >= 0 && static_cast<uint64_t>(value) <= kUint32Max, name, " exceeds uint32 range: ", value);
}

}  // namespace

at::Tensor fused_timestamp_lru_metadata_update(
    at::Tensor &slot_map, const at::Tensor &req_indices, const at::Tensor &topk_indices,
    const at::Tensor &device_token_pos, at::Tensor &device_lru_slots, at::Tensor &device_lru_slot_stamps,
    at::Tensor &device_slot_tokens, int64_t max_context_len, int64_t stamp_max, int64_t block_dim)
{
    CheckInt32Contiguous(slot_map, "slot_map");
    CheckInt32Contiguous(req_indices, "req_indices");
    CheckInt32Contiguous(topk_indices, "topk_indices");
    CheckInt32Contiguous(device_token_pos, "device_token_pos");
    CheckInt32Contiguous(device_lru_slots, "device_lru_slots");
    CheckInt32Contiguous(device_lru_slot_stamps, "device_lru_slot_stamps");
    CheckInt32Contiguous(device_slot_tokens, "device_slot_tokens");

    CheckSameDevice(req_indices, slot_map, "req_indices");
    CheckSameDevice(topk_indices, slot_map, "topk_indices");
    CheckSameDevice(device_token_pos, slot_map, "device_token_pos");
    CheckSameDevice(device_lru_slots, slot_map, "device_lru_slots");
    CheckSameDevice(device_lru_slot_stamps, slot_map, "device_lru_slot_stamps");
    CheckSameDevice(device_slot_tokens, slot_map, "device_slot_tokens");

    TORCH_CHECK(slot_map.dim() == 2, "slot_map must be 2-D");
    TORCH_CHECK(req_indices.dim() == 1, "req_indices must be 1-D");
    TORCH_CHECK(topk_indices.dim() == 2, "topk_indices must be 2-D");
    TORCH_CHECK(device_token_pos.dim() == 2, "device_token_pos must be 2-D");
    TORCH_CHECK(device_lru_slots.dim() == 2, "device_lru_slots must be 2-D");
    TORCH_CHECK(device_lru_slot_stamps.dim() == 2, "device_lru_slot_stamps must be 2-D");
    TORCH_CHECK(device_slot_tokens.dim() == 2, "device_slot_tokens must be 2-D");

    const int64_t batchSize64 = req_indices.size(0);
    const int64_t requestRows64 = device_lru_slots.size(0);
    const int64_t cacheCapacity64 = device_lru_slots.size(1);
    const int64_t topk64 = topk_indices.size(1);
    const int64_t slotMapRows64 = slot_map.size(0);
    const int64_t slotMapWidth64 = slot_map.size(1);

    TORCH_CHECK(batchSize64 > 0, "batch size must be positive");
    TORCH_CHECK(topk64 == kFixedTopk, "fused timestamp LRU requires topk=", kFixedTopk, ", got ", topk64);
    TORCH_CHECK(cacheCapacity64 == kFixedCacheCapacity,
                "fused timestamp LRU requires cache capacity=", kFixedCacheCapacity, ", got ", cacheCapacity64);
    TORCH_CHECK(topk_indices.size(0) == batchSize64, "topk_indices dim0 must match req_indices");
    TORCH_CHECK(device_token_pos.sizes() == topk_indices.sizes(),
                "device_token_pos shape must match topk_indices");
    TORCH_CHECK(device_lru_slot_stamps.sizes() == device_lru_slots.sizes(),
                "device_lru_slot_stamps shape must match device_lru_slots");
    TORCH_CHECK(device_slot_tokens.sizes() == device_lru_slots.sizes(),
                "device_slot_tokens shape must match device_lru_slots");
    TORCH_CHECK(slotMapRows64 >= requestRows64,
                "slot_map must have at least as many rows as device LRU state");
    TORCH_CHECK(max_context_len > 0 && max_context_len <= slotMapWidth64,
                "max_context_len must be in (0, slot_map.size(1)], got ", max_context_len);
    TORCH_CHECK(slotMapWidth64 % kAlignment == 0,
                "slot_map row width must be a multiple of ", kAlignment, ", got ", slotMapWidth64);
    TORCH_CHECK(stamp_max > 0 && stamp_max <= kMaxExactFp32Stamp,
                "stamp_max must be in [1, ", kMaxExactFp32Stamp, "] so float32 sort keys remain exact, got ",
                stamp_max);
    TORCH_CHECK(block_dim >= 0, "block_dim must be non-negative, got ", block_dim);

    CheckFitsUint32(batchSize64, "batch size");
    CheckFitsUint32(requestRows64, "request rows");
    CheckFitsUint32(slotMapRows64, "slot_map rows");
    CheckFitsUint32(slotMapWidth64, "slot_map width");
    CheckFitsUint32(max_context_len, "max_context_len");
    CheckFitsUint32(stamp_max, "stamp_max");
    TORCH_CHECK(static_cast<uint64_t>(slotMapRows64) * static_cast<uint64_t>(slotMapWidth64) <= kUint32Max,
                "slot_map storage exceeds the kernel uint32 address range");
    TORCH_CHECK(static_cast<uint64_t>(requestRows64) * kFixedCacheCapacity <= kUint32Max,
                "device metadata storage exceeds the kernel uint32 address range");
    TORCH_CHECK(static_cast<uint64_t>(batchSize64) * kFixedTopk <= kUint32Max,
                "batch output storage exceeds the kernel uint32 address range");

    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const uint32_t maxAivCoreNum = static_cast<uint32_t>(platform->GetCoreNumAiv());
    TORCH_CHECK(maxAivCoreNum > 0, "failed to get the available AIV core count");
    uint64_t ubSize = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    TORCH_CHECK(ubSize >= static_cast<uint64_t>(kRequiredWorkUbBytes + kPipeReserveBytes),
                "fused timestamp LRU requires at least ", kRequiredWorkUbBytes + kPipeReserveBytes,
                " bytes of UB, got ", ubSize);
    const uint32_t usableUbBytes = static_cast<uint32_t>(ubSize - kPipeReserveBytes);

    const uint32_t batchSize = static_cast<uint32_t>(batchSize64);
    const uint32_t requestRows = static_cast<uint32_t>(requestRows64);
    const uint32_t slotMapRows = static_cast<uint32_t>(slotMapRows64);
    const uint32_t slotMapWidth = static_cast<uint32_t>(slotMapWidth64);
    const uint32_t maxContextLen = static_cast<uint32_t>(max_context_len);
    const uint32_t stampMax = static_cast<uint32_t>(stamp_max);
    uint32_t effectiveBlockDim = block_dim > 0 ? static_cast<uint32_t>(block_dim)
                                               : std::min(batchSize, maxAivCoreNum);
    effectiveBlockDim = std::max(effectiveBlockDim, 1U);
    TORCH_CHECK(effectiveBlockDim <= maxAivCoreNum, "block_dim must not exceed available AIV cores ",
                maxAivCoreNum, ", got ", effectiveBlockDim);

    auto victimSlots = at::empty_like(topk_indices);
    auto npuStream = c10_npu::getCurrentNPUStream();
    slot_map.record_stream(npuStream);
    req_indices.record_stream(npuStream);
    topk_indices.record_stream(npuStream);
    device_token_pos.record_stream(npuStream);
    device_lru_slots.record_stream(npuStream);
    device_lru_slot_stamps.record_stream(npuStream);
    device_slot_tokens.record_stream(npuStream);
    victimSlots.record_stream(npuStream);

    EXEC_KERNEL_CMD(fused_timestamp_lru_metadata_update, effectiveBlockDim, slot_map, req_indices, topk_indices,
                    device_token_pos, device_lru_slots, device_lru_slot_stamps, device_slot_tokens, victimSlots,
                    batchSize, requestRows, slotMapRows, slotMapWidth, maxContextLen, stampMax, usableUbBytes);
    return victimSlots;
}

}  // namespace npu_kernel
}  // namespace sglang
