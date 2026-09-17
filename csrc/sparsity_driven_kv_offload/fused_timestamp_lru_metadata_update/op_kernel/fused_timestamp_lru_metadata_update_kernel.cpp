// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// A2/A3 AIV implementation of timestamp LRU victim selection and state update.
// The kernel intentionally uses no scatter instruction. Dense transforms,
// sorting, prefix scan, and indexed reads are SIMD operations. Sparse cache
// metadata writes run in the following parallel_lru_metadata_write kernel.

#include "kernel_operator.h"

namespace {

constexpr uint32_t kTopk = 2048;
constexpr uint32_t kCacheCapacity = 4096;
constexpr uint32_t kRecordCount = kCacheCapacity;
constexpr uint32_t kBytesPerInt = sizeof(int32_t);
constexpr uint32_t kSortPairElements = 2;
constexpr uint32_t kScalarBlockElements = 32 / kBytesPerInt;

// Stage A (4096-record stable hit partition) UB layout.
constexpr uint32_t kRecordValueOffset = 0;
constexpr uint32_t kRecordIndexOffset = kRecordValueOffset + kRecordCount * kBytesPerInt;
constexpr uint32_t kRecordSortTmpOffset = kRecordIndexOffset + kRecordCount * kBytesPerInt;
constexpr uint32_t kRecordSortOutOffset = kRecordSortTmpOffset + kSortPairElements * kRecordCount * kBytesPerInt;
constexpr uint32_t kLruSlotsOffset = kRecordSortOutOffset + kSortPairElements * kRecordCount * kBytesPerInt;
constexpr uint32_t kLruStampsOffset = kLruSlotsOffset + kCacheCapacity * kBytesPerInt;
constexpr uint32_t kPositionMaskOffset = kLruStampsOffset + kCacheCapacity * kBytesPerInt;
constexpr uint32_t kWorkUbBytes = kPositionMaskOffset + kCacheCapacity * kBytesPerInt;

// The sort output is dead immediately after Extract, so keep the sorted LRU
// pair there for the victim-plan and writeback stages.
constexpr uint32_t kSortedSlotsOffset = kRecordSortOutOffset;
constexpr uint32_t kSortedStampsOffset =
    kRecordSortOutOffset + kCacheCapacity * kBytesPerInt;

constexpr uint32_t kTopkTokenOffset = 0;
constexpr uint32_t kTopkDevicePosOffset = kTopk * kBytesPerInt;
constexpr uint32_t kMissFlagOffset = 2 * kTopk * kBytesPerInt;
constexpr uint32_t kScanScratchOffset = 3 * kTopk * kBytesPerInt;
constexpr uint32_t kVictimOffset = 4 * kTopk * kBytesPerInt;
constexpr uint32_t kGatherOffsetOffset = 5 * kTopk * kBytesPerInt;
constexpr uint32_t kVectorScratchOffset = 6 * kTopk * kBytesPerInt;
constexpr uint32_t kMissCountStagingOffset = kPositionMaskOffset;

static_assert(kWorkUbBytes == 147456, "unexpected UB layout size");

template <AscendC::HardEvent event>
__aicore__ inline void SyncPipes()
{
    const int32_t eventId = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventId);
    AscendC::WaitFlag<event>(eventId);
}

__aicore__ inline void SyncMte2ToVector()
{
    SyncPipes<AscendC::HardEvent::MTE2_V>();
}

__aicore__ inline void SyncVectorToMte3()
{
    SyncPipes<AscendC::HardEvent::V_MTE3>();
}

__aicore__ inline void SyncVectorToScalar()
{
    SyncPipes<AscendC::HardEvent::V_S>();
}

__aicore__ inline void SyncMte3ToScalar()
{
    SyncPipes<AscendC::HardEvent::MTE3_S>();
}

__aicore__ inline void SyncScalarToMte3()
{
    SyncPipes<AscendC::HardEvent::S_MTE3>();
}

template <typename T>
__aicore__ inline void CopyRowIn(const AscendC::LocalTensor<T> &dst, const AscendC::GlobalTensor<T> &src,
                                 uint32_t elementCount)
{
    const uint32_t byteCount =
        elementCount * static_cast<uint32_t>(sizeof(T));
    AscendC::DataCopyExtParams copyParams{1, byteCount, 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    AscendC::DataCopyPad(dst, src, copyParams, padParams);
}

template <typename T>
__aicore__ inline void CopyRowOut(const AscendC::GlobalTensor<T> &dst, const AscendC::LocalTensor<T> &src,
                                  uint32_t elementCount)
{
    if (elementCount == 0) {
        return;
    }
    const uint32_t byteCount =
        elementCount * static_cast<uint32_t>(sizeof(T));
    AscendC::DataCopyExtParams copyParams{1, byteCount, 0, 0, 0};
    AscendC::DataCopyPad(dst, src, copyParams);
}

class KernelFusedTimestampLruMetadataUpdate
{
public:
    __aicore__ inline KernelFusedTimestampLruMetadataUpdate() {}

    __aicore__ inline void Init(GM_ADDR reqIndices, GM_ADDR topkIndices,
                                GM_ADDR deviceTokenPos, GM_ADDR hitPositionMask,
                                GM_ADDR deviceLruSlots, GM_ADDR deviceLruSlotStamps,
                                GM_ADDR victimSlots, GM_ADDR missCounts,
                                uint32_t batchSize, uint32_t requestRows,
                                uint32_t maxContextLen, uint32_t stampMax, uint32_t usableUbBytes,
                                AscendC::TPipe *pipe)
    {
        this->batchSize = batchSize;
        this->requestRows = requestRows;
        this->maxContextLen = maxContextLen;
        this->stampMax = stampMax;

        reqIndicesGm.SetGlobalBuffer((__gm__ int32_t *)reqIndices, batchSize);
        topkIndicesGm.SetGlobalBuffer((__gm__ int32_t *)topkIndices,
                                     static_cast<uint64_t>(batchSize) * kTopk);
        deviceTokenPosGm.SetGlobalBuffer((__gm__ int32_t *)deviceTokenPos,
                                        static_cast<uint64_t>(batchSize) * kTopk);
        hitPositionMaskGm.SetGlobalBuffer((__gm__ int32_t *)hitPositionMask,
                                         static_cast<uint64_t>(batchSize) * kCacheCapacity);
        deviceLruSlotsGm.SetGlobalBuffer((__gm__ int32_t *)deviceLruSlots,
                                        static_cast<uint64_t>(requestRows) * kCacheCapacity);
        deviceLruSlotStampsGm.SetGlobalBuffer((__gm__ int32_t *)deviceLruSlotStamps,
                                             static_cast<uint64_t>(requestRows) * kCacheCapacity);
        victimSlotsGm.SetGlobalBuffer((__gm__ int32_t *)victimSlots,
                                     static_cast<uint64_t>(batchSize) * kTopk);
        missCountsGm.SetGlobalBuffer((__gm__ int32_t *)missCounts, batchSize);

        // The host obtains this value from the current platform and verifies
        // that the fixed A2/A3 memory plan fits before launching the kernel.
        pipe->InitBuffer(workBuf, usableUbBytes);
    }

    __aicore__ inline void Process()
    {
        const uint32_t coreIdx = AscendC::GetBlockIdx();
        const uint32_t coreCount = AscendC::GetBlockNum();
        for (uint32_t batchIdx = coreIdx; batchIdx < batchSize; batchIdx += coreCount) {
            ProcessRequest(batchIdx);
        }
    }

private:
    __aicore__ inline void ProcessRequest(uint32_t batchIdx)
    {
        const int32_t reqId = reqIndicesGm.GetValue(batchIdx);
        // Negative sentinel and out-of-range request rows are masked by the
        // caller. Leave their victim output undefined and skip all GM writes.
        if (reqId < 0 || static_cast<uint32_t>(reqId) >= requestRows) {
            return;
        }

        const uint32_t requestRow = static_cast<uint32_t>(reqId);
        BuildUpdatedSlotStampPairs(batchIdx, requestRow);
        const uint32_t missCount = BuildVictimPlan(batchIdx);
        WriteMissCount(batchIdx, missCount);
        WriteRotatedLruState(requestRow, missCount);
    }

    // The persistent LRU pairs are already ordered by descending timestamp.
    // Stable-partition them into non-hits followed by hits. Preserving the old
    // order inside each group preserves timestamp order, while hit timestamps
    // are reset to zero. Unique keys make the result independent of Sort's tie
    // behavior: nonHit * C + (C - 1 - oldIndex), sorted descending.
    __aicore__ inline void BuildUpdatedSlotStampPairs(uint32_t batchIdx, uint32_t requestRow)
    {
        AscendC::LocalTensor<float> recordValue =
            workBuf.GetWithOffset<float>(kRecordCount, kRecordValueOffset);
        AscendC::LocalTensor<int32_t> recordIndex =
            workBuf.GetWithOffset<int32_t>(kRecordCount, kRecordIndexOffset);
        AscendC::LocalTensor<float> sortTmp =
            workBuf.GetWithOffset<float>(2 * kRecordCount, kRecordSortTmpOffset);
        AscendC::LocalTensor<float> sortOut =
            workBuf.GetWithOffset<float>(2 * kRecordCount, kRecordSortOutOffset);
        AscendC::LocalTensor<int32_t> lruSlots =
            workBuf.GetWithOffset<int32_t>(kCacheCapacity, kLruSlotsOffset);
        AscendC::LocalTensor<int32_t> lruStamps =
            workBuf.GetWithOffset<int32_t>(kCacheCapacity, kLruStampsOffset);
        AscendC::LocalTensor<int32_t> positionMask =
            workBuf.GetWithOffset<int32_t>(kCacheCapacity, kPositionMaskOffset);
        AscendC::LocalTensor<int32_t> gatherOffsets =
            workBuf.GetWithOffset<int32_t>(kCacheCapacity, kRecordSortTmpOffset);
        AscendC::LocalTensor<int32_t> nonHit =
            workBuf.GetWithOffset<int32_t>(kCacheCapacity, kRecordSortOutOffset);

        CopyRowIn(lruSlots, deviceLruSlotsGm[requestRow * kCacheCapacity], kCacheCapacity);
        CopyRowIn(lruStamps, deviceLruSlotStampsGm[requestRow * kCacheCapacity], kCacheCapacity);
        CopyRowIn(positionMask, hitPositionMaskGm[batchIdx * kCacheCapacity], kCacheCapacity);
        SyncMte2ToVector();

        // Saturating age increment: min(stamp, stampMax - 1) + 1.
        AscendC::Mins(lruStamps, lruStamps, static_cast<int32_t>(stampMax - 1), kCacheCapacity);
        AscendC::Adds(lruStamps, lruStamps, static_cast<int32_t>(1), kCacheCapacity);

        // Convert physical slots to byte offsets and gather the hit bit in
        // current LRU order. positionMask is guaranteed to contain only 0/1.
        AscendC::Muls(gatherOffsets, lruSlots, static_cast<int32_t>(kBytesPerInt), kCacheCapacity);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Gather(nonHit, positionMask, gatherOffsets.ReinterpretCast<uint32_t>(),
                        static_cast<uint32_t>(0), kCacheCapacity);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(nonHit, nonHit, static_cast<int32_t>(-1), kCacheCapacity);
        AscendC::Adds(nonHit, nonHit, static_cast<int32_t>(1), kCacheCapacity);
        AscendC::Mul(lruStamps, lruStamps, nonHit, kCacheCapacity);

        // Build exact, unique stable-partition keys in [0, 8191].
        AscendC::Muls(nonHit, nonHit, static_cast<int32_t>(kCacheCapacity), kCacheCapacity);
        AscendC::CreateVecIndex(recordIndex, static_cast<int32_t>(0), kCacheCapacity);
        AscendC::Muls(gatherOffsets, recordIndex, static_cast<int32_t>(-1), kCacheCapacity);
        AscendC::Adds(gatherOffsets, gatherOffsets,
                      static_cast<int32_t>(kCacheCapacity - 1), kCacheCapacity);
        AscendC::Add(nonHit, nonHit, gatherOffsets, kCacheCapacity);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(recordValue, nonHit, AscendC::RoundMode::CAST_NONE, kCacheCapacity);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Sort<float, true>(sortOut, recordValue, recordIndex.ReinterpretCast<uint32_t>(), sortTmp,
                                   kRecordCount / 32);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Extract(recordValue, recordIndex.ReinterpretCast<uint32_t>(), sortOut, kRecordCount / 32);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::LocalTensor<int32_t> sortedSlots =
            workBuf.GetWithOffset<int32_t>(kCacheCapacity, kSortedSlotsOffset);
        AscendC::LocalTensor<int32_t> sortedStamps =
            workBuf.GetWithOffset<int32_t>(kCacheCapacity, kSortedStampsOffset);
        AscendC::Muls(gatherOffsets, recordIndex, static_cast<int32_t>(kBytesPerInt), kCacheCapacity);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Gather(sortedSlots, lruSlots, gatherOffsets.ReinterpretCast<uint32_t>(),
                        static_cast<uint32_t>(0), kCacheCapacity);
        AscendC::Gather(sortedStamps, lruStamps, gatherOffsets.ReinterpretCast<uint32_t>(),
                        static_cast<uint32_t>(0), kCacheCapacity);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline uint32_t BuildVictimPlan(uint32_t batchIdx)
    {
        AscendC::LocalTensor<int32_t> topkTokens =
            workBuf.GetWithOffset<int32_t>(kTopk, kTopkTokenOffset);
        AscendC::LocalTensor<int32_t> devicePos =
            workBuf.GetWithOffset<int32_t>(kTopk, kTopkDevicePosOffset);
        AscendC::LocalTensor<int32_t> missFlag =
            workBuf.GetWithOffset<int32_t>(kTopk, kMissFlagOffset);
        AscendC::LocalTensor<int32_t> scanScratch =
            workBuf.GetWithOffset<int32_t>(kTopk, kScanScratchOffset);
        AscendC::LocalTensor<int32_t> victims =
            workBuf.GetWithOffset<int32_t>(kTopk, kVictimOffset);
        AscendC::LocalTensor<int32_t> gatherOffsets =
            workBuf.GetWithOffset<int32_t>(kTopk, kGatherOffsetOffset);
        AscendC::LocalTensor<int32_t> vectorScratch =
            workBuf.GetWithOffset<int32_t>(kTopk, kVectorScratchOffset);
        AscendC::LocalTensor<int32_t> sortedSlots =
            workBuf.GetWithOffset<int32_t>(kCacheCapacity, kSortedSlotsOffset);

        CopyRowIn(topkTokens, topkIndicesGm[batchIdx * kTopk], kTopk);
        CopyRowIn(devicePos, deviceTokenPosGm[batchIdx * kTopk], kTopk);
        SyncMte2ToVector();

        // missFlag = valid_topk && (device_pos == -1), expressed entirely as
        // clamped int32 SIMD arithmetic.
        AscendC::Adds(missFlag, devicePos, static_cast<int32_t>(1), kTopk);
        AscendC::Maxs(missFlag, missFlag, static_cast<int32_t>(0), kTopk);
        AscendC::Mins(missFlag, missFlag, static_cast<int32_t>(1), kTopk);
        AscendC::Duplicate(scanScratch, static_cast<int32_t>(1), kTopk);
        AscendC::Sub(missFlag, scanScratch, missFlag, kTopk);

        AscendC::Adds(scanScratch, topkTokens, static_cast<int32_t>(1), kTopk);
        AscendC::Maxs(scanScratch, scanScratch, static_cast<int32_t>(0), kTopk);
        AscendC::Mins(scanScratch, scanScratch, static_cast<int32_t>(1), kTopk);
        AscendC::Duplicate(vectorScratch, static_cast<int32_t>(maxContextLen), kTopk);
        AscendC::Sub(vectorScratch, vectorScratch, topkTokens, kTopk);
        AscendC::Maxs(vectorScratch, vectorScratch, static_cast<int32_t>(0), kTopk);
        AscendC::Mins(vectorScratch, vectorScratch, static_cast<int32_t>(1), kTopk);
        AscendC::Min(scanScratch, scanScratch, vectorScratch, kTopk);
        AscendC::Min(missFlag, missFlag, scanScratch, kTopk);
        AscendC::PipeBarrier<PIPE_V>();

        // Hillis-Steele inclusive scan. Gather supplies a zero-padded shifted
        // vector on platforms where scatter is unavailable.
        for (uint32_t step = 1; step < kTopk; step <<= 1) {
            AscendC::CreateVecIndex(gatherOffsets, static_cast<int32_t>(0), kTopk);
            AscendC::Adds(gatherOffsets, gatherOffsets, -static_cast<int32_t>(step), kTopk);
            AscendC::Maxs(gatherOffsets, gatherOffsets, static_cast<int32_t>(0), kTopk);
            AscendC::Muls(gatherOffsets, gatherOffsets, static_cast<int32_t>(kBytesPerInt), kTopk);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Gather(scanScratch, missFlag, gatherOffsets.ReinterpretCast<uint32_t>(),
                            static_cast<uint32_t>(0), kTopk);
            AscendC::Duplicate(scanScratch, static_cast<int32_t>(0), step);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(missFlag, missFlag, scanScratch, kTopk);
            AscendC::PipeBarrier<PIPE_V>();
        }

        SyncVectorToScalar();
        const uint32_t missCount = static_cast<uint32_t>(missFlag.GetValue(kTopk - 1));

        // Gather the r-th oldest eligible slot for the r-th miss, then restore
        // -1 in hit/invalid positions without a scatter operation.
        AscendC::Adds(gatherOffsets, missFlag, static_cast<int32_t>(-1), kTopk);
        AscendC::Maxs(gatherOffsets, gatherOffsets, static_cast<int32_t>(0), kTopk);
        AscendC::Muls(gatherOffsets, gatherOffsets, static_cast<int32_t>(kBytesPerInt), kTopk);
        AscendC::Gather(victims, sortedSlots, gatherOffsets.ReinterpretCast<uint32_t>(),
                        static_cast<uint32_t>(0), kTopk);
        AscendC::PipeBarrier<PIPE_V>();

        // Recover the original 0/1 miss vector as scan[i] - scan[i-1].
        AscendC::CreateVecIndex(gatherOffsets, static_cast<int32_t>(0), kTopk);
        AscendC::Adds(gatherOffsets, gatherOffsets, static_cast<int32_t>(-1), kTopk);
        AscendC::Maxs(gatherOffsets, gatherOffsets, static_cast<int32_t>(0), kTopk);
        AscendC::Muls(gatherOffsets, gatherOffsets, static_cast<int32_t>(kBytesPerInt), kTopk);
        AscendC::Gather(scanScratch, missFlag, gatherOffsets.ReinterpretCast<uint32_t>(),
                        static_cast<uint32_t>(0), kTopk);
        AscendC::Duplicate(scanScratch, static_cast<int32_t>(0), 1);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(scanScratch, missFlag, scanScratch, kTopk);
        AscendC::Adds(victims, victims, static_cast<int32_t>(1), kTopk);
        AscendC::Mul(victims, victims, scanScratch, kTopk);
        AscendC::Adds(victims, victims, static_cast<int32_t>(-1), kTopk);
        AscendC::PipeBarrier<PIPE_V>();

        SyncVectorToMte3();
        CopyRowOut(victimSlotsGm[batchIdx * kTopk], victims, kTopk);
        return missCount;
    }

    __aicore__ inline void WriteMissCount(uint32_t batchIdx, uint32_t missCount)
    {
        AscendC::LocalTensor<int32_t> staging =
            workBuf.GetWithOffset<int32_t>(kScalarBlockElements, kMissCountStagingOffset);
        staging.SetValue(0, static_cast<int32_t>(missCount));
        SyncScalarToMte3();
        AscendC::DataCopyExtParams oneIntParams{1, sizeof(int32_t), 0, 0, 0};
        AscendC::DataCopyPad(missCountsGm[batchIdx], staging, oneIntParams);
    }

    __aicore__ inline void WriteRotatedLruState(uint32_t requestRow, uint32_t missCount)
    {
        AscendC::LocalTensor<int32_t> sortedSlots =
            workBuf.GetWithOffset<int32_t>(kCacheCapacity, kSortedSlotsOffset);
        AscendC::LocalTensor<int32_t> sortedStamps =
            workBuf.GetWithOffset<int32_t>(kCacheCapacity, kSortedStampsOffset);
        AscendC::LocalTensor<int32_t> gatherOffsets =
            workBuf.GetWithOffset<int32_t>(kCacheCapacity, kTopkTokenOffset);
        AscendC::LocalTensor<int32_t> wrapFlags =
            workBuf.GetWithOffset<int32_t>(kCacheCapacity, kMissFlagOffset);
        AscendC::LocalTensor<int32_t> rotatedSlots =
            workBuf.GetWithOffset<int32_t>(kCacheCapacity, kLruSlotsOffset);
        AscendC::LocalTensor<int32_t> rotatedStamps =
            workBuf.GetWithOffset<int32_t>(kCacheCapacity, kLruStampsOffset);

        // Newly occupied victims become MRU (stamp zero). Rotating them to the
        // tail keeps the persistent pair array in descending stamp order.
        if (missCount > 0) {
            AscendC::Duplicate(sortedStamps, static_cast<int32_t>(0), missCount);
            AscendC::PipeBarrier<PIPE_V>();
        }

        // Build byte offsets for (i + missCount) % capacity and gather into
        // aligned full-row buffers. This avoids DataCopyPad sources at
        // sortedSlots[missCount]/sortedStamps[missCount], whose runtime address
        // is generally only 4-byte aligned.
        AscendC::CreateVecIndex(gatherOffsets, static_cast<int32_t>(missCount), kCacheCapacity);
        AscendC::Adds(wrapFlags, gatherOffsets, -static_cast<int32_t>(kCacheCapacity - 1), kCacheCapacity);
        AscendC::Maxs(wrapFlags, wrapFlags, static_cast<int32_t>(0), kCacheCapacity);
        AscendC::Mins(wrapFlags, wrapFlags, static_cast<int32_t>(1), kCacheCapacity);
        AscendC::Muls(wrapFlags, wrapFlags, static_cast<int32_t>(kCacheCapacity), kCacheCapacity);
        AscendC::Sub(gatherOffsets, gatherOffsets, wrapFlags, kCacheCapacity);
        AscendC::Muls(gatherOffsets, gatherOffsets, static_cast<int32_t>(kBytesPerInt), kCacheCapacity);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Gather(rotatedSlots, sortedSlots, gatherOffsets.ReinterpretCast<uint32_t>(),
                        static_cast<uint32_t>(0), kCacheCapacity);
        AscendC::Gather(rotatedStamps, sortedStamps, gatherOffsets.ReinterpretCast<uint32_t>(),
                        static_cast<uint32_t>(0), kCacheCapacity);
        AscendC::PipeBarrier<PIPE_V>();
        SyncVectorToMte3();

        const uint32_t rowOffset = requestRow * kCacheCapacity;
        CopyRowOut(deviceLruSlotsGm[rowOffset], rotatedSlots, kCacheCapacity);
        CopyRowOut(deviceLruSlotStampsGm[rowOffset], rotatedStamps, kCacheCapacity);
        // Drain both full-row writes before this core reuses the shared UB for
        // the next grid-stride request or exits the kernel.
        SyncMte3ToScalar();
    }

private:
    AscendC::TBuf<AscendC::TPosition::VECCALC> workBuf;
    AscendC::GlobalTensor<int32_t> reqIndicesGm;
    AscendC::GlobalTensor<int32_t> topkIndicesGm;
    AscendC::GlobalTensor<int32_t> deviceTokenPosGm;
    AscendC::GlobalTensor<int32_t> hitPositionMaskGm;
    AscendC::GlobalTensor<int32_t> deviceLruSlotsGm;
    AscendC::GlobalTensor<int32_t> deviceLruSlotStampsGm;
    AscendC::GlobalTensor<int32_t> victimSlotsGm;
    AscendC::GlobalTensor<int32_t> missCountsGm;

    uint32_t batchSize = 0;
    uint32_t requestRows = 0;
    uint32_t maxContextLen = 0;
    uint32_t stampMax = 0;
};

}  // namespace

extern "C" __global__ __aicore__ void fused_timestamp_lru_metadata_update(
    GM_ADDR req_indices, GM_ADDR topk_indices, GM_ADDR device_token_pos,
    GM_ADDR hit_position_mask, GM_ADDR device_lru_slots, GM_ADDR device_lru_slot_stamps,
    GM_ADDR victim_slots, GM_ADDR miss_counts, uint32_t batch_size, uint32_t request_rows,
    uint32_t max_context_len, uint32_t stamp_max, uint32_t usable_ub_bytes)
{
    AscendC::TPipe pipe;
    KernelFusedTimestampLruMetadataUpdate kernel;
    kernel.Init(req_indices, topk_indices, device_token_pos, hit_position_mask, device_lru_slots,
                device_lru_slot_stamps, victim_slots, miss_counts, batch_size, request_rows,
                max_context_len, stamp_max, usable_ub_bytes, &pipe);
    kernel.Process();
}
