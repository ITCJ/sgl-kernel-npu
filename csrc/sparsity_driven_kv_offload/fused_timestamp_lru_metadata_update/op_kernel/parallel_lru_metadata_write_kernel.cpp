// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Parallel sparse metadata writes for timestamp-LRU refill. The preceding
// victim-selection kernel produces victim_slots and miss_counts on the same
// stream. Each request is divided into 64 independent 32-entry tiles so one
// request can use many AIVs when it contains many misses.

#include "kernel_operator.h"

namespace {

constexpr uint32_t kTopk = 2048;
constexpr uint32_t kCacheCapacity = 4096;
constexpr uint32_t kTileElements = 32;
constexpr uint32_t kTilesPerBatch = kTopk / kTileElements;
constexpr uint32_t kBytesPerInt = sizeof(int32_t);
constexpr uint32_t kScalarBlockElements = 32 / kBytesPerInt;
constexpr uint32_t kSlotLineElements = kScalarBlockElements;
constexpr uint32_t kSlotLineCount = kTileElements + 1;
constexpr uint32_t kSentinelLineOffset = kTileElements * kSlotLineElements;
constexpr uint32_t kSentinelLineOffsetBytes = kSentinelLineOffset * kBytesPerInt;
constexpr uint32_t kSparseWriteBatch = 8;
constexpr uint32_t kInvalidSparseOffset = 0xFFFFFFFFU;
static_assert(kTopk % kTileElements == 0, "top-k must divide into complete metadata tiles");

template <AscendC::HardEvent event>
__aicore__ inline void SyncPipes()
{
    const int32_t eventId = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventId);
    AscendC::WaitFlag<event>(eventId);
}

class KernelParallelLruMetadataWrite
{
public:
    __aicore__ inline KernelParallelLruMetadataWrite() {}

    __aicore__ inline void Init(GM_ADDR slotMap, GM_ADDR reqIndices, GM_ADDR topkIndices,
                                GM_ADDR victimSlots, GM_ADDR missCounts, GM_ADDR deviceSlotTokens,
                                uint32_t batchSize, uint32_t requestRows, uint32_t slotMapWidth,
                                uint32_t maxContextLen, AscendC::TPipe *pipe)
    {
        this->batchSize = batchSize;
        this->requestRows = requestRows;
        this->slotMapWidth = slotMapWidth;
        this->maxContextLen = maxContextLen;

        slotMapGm.SetGlobalBuffer((__gm__ int32_t *)slotMap,
                                  static_cast<uint64_t>(requestRows) * slotMapWidth);
        reqIndicesGm.SetGlobalBuffer((__gm__ int32_t *)reqIndices, batchSize);
        topkIndicesGm.SetGlobalBuffer((__gm__ int32_t *)topkIndices,
                                     static_cast<uint64_t>(batchSize) * kTopk);
        victimSlotsGm.SetGlobalBuffer((__gm__ int32_t *)victimSlots,
                                     static_cast<uint64_t>(batchSize) * kTopk);
        missCountsGm.SetGlobalBuffer((__gm__ int32_t *)missCounts, batchSize);
        deviceSlotTokensGm.SetGlobalBuffer((__gm__ int32_t *)deviceSlotTokens,
                                          static_cast<uint64_t>(requestRows) * kCacheCapacity);

        pipe->InitBuffer(topkBuf, kTileElements * kBytesPerInt);
        pipe->InitBuffer(victimBuf, kTileElements * kBytesPerInt);
        pipe->InitBuffer(oldTokenLineBuf,
                         kSlotLineCount * kSlotLineElements * kBytesPerInt);
        pipe->InitBuffer(oldTokenOffsetBuf, kTileElements * kBytesPerInt);
        pipe->InitBuffer(oldTokenBuf, kTileElements * kBytesPerInt);
        pipe->InitBuffer(minusOneBuf, 32);
        pipe->InitBuffer(newTokenStagingBuf,
                         kSparseWriteBatch * kScalarBlockElements * kBytesPerInt);
        pipe->InitBuffer(victimStagingBuf,
                         kSparseWriteBatch * kScalarBlockElements * kBytesPerInt);
    }

    __aicore__ inline void Process()
    {
        AscendC::LocalTensor<int32_t> minusOne = minusOneBuf.Get<int32_t>();
        AscendC::LocalTensor<int32_t> oldTokenLines =
            oldTokenLineBuf.Get<int32_t>();
        AscendC::Duplicate(minusOne, static_cast<int32_t>(-1), kScalarBlockElements);
        AscendC::Duplicate(oldTokenLines[kSentinelLineOffset],
                           static_cast<int32_t>(-1), kSlotLineElements);
        SyncPipes<AscendC::HardEvent::V_MTE3>();

        const uint32_t workerIdx = AscendC::GetBlockIdx();
        const uint32_t workerNum = AscendC::GetBlockNum();
        const uint32_t taskCount = batchSize * kTilesPerBatch;
        for (uint32_t task = workerIdx; task < taskCount; task += workerNum) {
            ProcessTile(task, minusOne);
        }
    }

private:
    __aicore__ inline void ProcessTile(
        uint32_t task, const AscendC::LocalTensor<int32_t> &minusOne)
    {
        const uint32_t batchIdx = task / kTilesPerBatch;
        const int32_t reqId = reqIndicesGm.GetValue(batchIdx);
        if (reqId < 0 || static_cast<uint32_t>(reqId) >= requestRows ||
            missCountsGm.GetValue(batchIdx) == 0) {
            return;
        }

        const uint32_t requestRow = static_cast<uint32_t>(reqId);
        const uint32_t tileIdx = task - batchIdx * kTilesPerBatch;
        const uint32_t gmOffset = batchIdx * kTopk + tileIdx * kTileElements;
        AscendC::LocalTensor<int32_t> topkLocal = topkBuf.Get<int32_t>();
        AscendC::LocalTensor<int32_t> victimsLocal = victimBuf.Get<int32_t>();
        AscendC::LocalTensor<int32_t> oldTokenLines =
            oldTokenLineBuf.Get<int32_t>();
        AscendC::LocalTensor<uint32_t> oldTokenOffsets =
            oldTokenOffsetBuf.Get<uint32_t>();
        AscendC::LocalTensor<int32_t> oldTokens = oldTokenBuf.Get<int32_t>();
        AscendC::DataCopy(topkLocal, topkIndicesGm[gmOffset], kTileElements);
        AscendC::DataCopy(victimsLocal, victimSlotsGm[gmOffset], kTileElements);
        SyncPipes<AscendC::HardEvent::MTE2_S>();

        // Load one aligned 32-byte reverse-map line for each victim. Invalid
        // positions gather from the immutable -1 sentinel line.
        const uint32_t slotTokenRowOffset = requestRow * kCacheCapacity;
        for (uint32_t i = 0; i < kTileElements; ++i) {
            const int32_t victim = victimsLocal.GetValue(i);
            if (victim < 0 || static_cast<uint32_t>(victim) >= kCacheCapacity) {
                oldTokenOffsets.SetValue(i, kSentinelLineOffsetBytes);
                continue;
            }
            const uint32_t victimU = static_cast<uint32_t>(victim);
            const uint32_t gmLineBase = victimU & ~(kSlotLineElements - 1U);
            const uint32_t lineLane = victimU & (kSlotLineElements - 1U);
            const uint32_t ubLineBase = i * kSlotLineElements;
            oldTokenOffsets.SetValue(i, (ubLineBase + lineLane) * kBytesPerInt);
            AscendC::DataCopy(oldTokenLines[ubLineBase],
                              deviceSlotTokensGm[slotTokenRowOffset + gmLineBase],
                              kSlotLineElements);
        }
        SyncPipes<AscendC::HardEvent::S_V>();
        SyncPipes<AscendC::HardEvent::MTE2_V>();
        AscendC::Gather(oldTokens, oldTokenLines, oldTokenOffsets,
                        static_cast<uint32_t>(0), kTileElements);
        AscendC::PipeBarrier<PIPE_V>();
        SyncPipes<AscendC::HardEvent::V_S>();

        AscendC::LocalTensor<int32_t> newTokenStaging =
            newTokenStagingBuf.Get<int32_t>();
        AscendC::LocalTensor<int32_t> victimStaging =
            victimStagingBuf.Get<int32_t>();
        uint32_t oldMapOffsets[kSparseWriteBatch];
        uint32_t slotTokenOffsets[kSparseWriteBatch];
        uint32_t newMapOffsets[kSparseWriteBatch];
        uint32_t pendingWrites = 0;

        for (uint32_t i = 0; i < kTileElements; ++i) {
            const int32_t victim = victimsLocal.GetValue(i);
            if (victim < 0 || static_cast<uint32_t>(victim) >= kCacheCapacity) {
                continue;
            }
            const int32_t newToken = topkLocal.GetValue(i);
            if (newToken < 0 || static_cast<uint32_t>(newToken) >= maxContextLen) {
                continue;
            }

            const uint32_t victimU = static_cast<uint32_t>(victim);
            const uint32_t slotTokenOffset = slotTokenRowOffset + victimU;
            const int32_t oldToken = oldTokens.GetValue(i);
            if (oldToken >= 0 && static_cast<uint32_t>(oldToken) < maxContextLen) {
                oldMapOffsets[pendingWrites] =
                    requestRow * slotMapWidth + static_cast<uint32_t>(oldToken);
            } else {
                oldMapOffsets[pendingWrites] = kInvalidSparseOffset;
            }
            slotTokenOffsets[pendingWrites] = slotTokenOffset;
            newMapOffsets[pendingWrites] =
                requestRow * slotMapWidth + static_cast<uint32_t>(newToken);

            const uint32_t stagingIndex = pendingWrites * kScalarBlockElements;
            newTokenStaging.SetValue(stagingIndex, newToken);
            victimStaging.SetValue(stagingIndex, victim);
            ++pendingWrites;

            if (pendingWrites == kSparseWriteBatch) {
                FlushWrites(minusOne, newTokenStaging, victimStaging,
                            oldMapOffsets, slotTokenOffsets, newMapOffsets,
                            pendingWrites);
                pendingWrites = 0;
            }
        }

        if (pendingWrites > 0) {
            FlushWrites(minusOne, newTokenStaging, victimStaging,
                        oldMapOffsets, slotTokenOffsets, newMapOffsets,
                        pendingWrites);
        }
    }

    __aicore__ inline void FlushWrites(
        const AscendC::LocalTensor<int32_t> &minusOne,
        const AscendC::LocalTensor<int32_t> &newTokenStaging,
        const AscendC::LocalTensor<int32_t> &victimStaging,
        const uint32_t *oldMapOffsets, const uint32_t *slotTokenOffsets,
        const uint32_t *newMapOffsets, uint32_t writeCount)
    {
        SyncPipes<AscendC::HardEvent::S_MTE3>();
        AscendC::DataCopyExtParams oneIntParams{1, sizeof(int32_t), 0, 0, 0};
        for (uint32_t i = 0; i < writeCount; ++i) {
            if (oldMapOffsets[i] != kInvalidSparseOffset) {
                AscendC::DataCopyPad(slotMapGm[oldMapOffsets[i]], minusOne,
                                     oneIntParams);
            }
            const uint32_t stagingIndex = i * kScalarBlockElements;
            AscendC::DataCopyPad(deviceSlotTokensGm[slotTokenOffsets[i]],
                                 newTokenStaging[stagingIndex], oneIntParams);
            AscendC::DataCopyPad(slotMapGm[newMapOffsets[i]],
                                 victimStaging[stagingIndex], oneIntParams);
        }
        SyncPipes<AscendC::HardEvent::MTE3_S>();
    }

private:
    AscendC::TBuf<AscendC::TPosition::VECCALC> topkBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> victimBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> oldTokenLineBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> oldTokenOffsetBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> oldTokenBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> minusOneBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> newTokenStagingBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> victimStagingBuf;

    AscendC::GlobalTensor<int32_t> slotMapGm;
    AscendC::GlobalTensor<int32_t> reqIndicesGm;
    AscendC::GlobalTensor<int32_t> topkIndicesGm;
    AscendC::GlobalTensor<int32_t> victimSlotsGm;
    AscendC::GlobalTensor<int32_t> missCountsGm;
    AscendC::GlobalTensor<int32_t> deviceSlotTokensGm;

    uint32_t batchSize = 0;
    uint32_t requestRows = 0;
    uint32_t slotMapWidth = 0;
    uint32_t maxContextLen = 0;
};

}  // namespace

extern "C" __global__ __aicore__ void parallel_lru_metadata_write(
    GM_ADDR slot_map, GM_ADDR req_indices, GM_ADDR topk_indices,
    GM_ADDR victim_slots, GM_ADDR miss_counts, GM_ADDR device_slot_tokens,
    uint32_t batch_size, uint32_t request_rows, uint32_t slot_map_width,
    uint32_t max_context_len)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    KernelParallelLruMetadataWrite kernel;
    kernel.Init(slot_map, req_indices, topk_indices, victim_slots, miss_counts,
                device_slot_tokens, batch_size, request_rows, slot_map_width,
                max_context_len, &pipe);
    kernel.Process();
}
