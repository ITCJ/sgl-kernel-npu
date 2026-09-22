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

// Stage A (4096-record stable hit partition and optional merge) UB layout.
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
constexpr uint32_t kMissCountStagingOffset = kVectorScratchOffset;

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

__aicore__ inline void SyncScalarToVector()
{
    SyncPipes<AscendC::HardEvent::S_V>();
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
                                uint32_t maxContextLen, uint32_t stampMax, uint32_t probationAge,
                                uint32_t halveHitStamps,
                                uint32_t usableUbBytes,
                                AscendC::TPipe *pipe)
    {
        this->batchSize = batchSize;
        this->requestRows = requestRows;
        this->maxContextLen = maxContextLen;
        this->stampMax = stampMax;
        this->probationAge = probationAge;
        this->halveHitStamps = halveHitStamps;

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
        WriteLruState(requestRow, missCount);
    }

    // The persistent LRU pairs are already ordered by descending timestamp.
    // Stable-partition them into non-hits followed by hits. Each group remains
    // sorted because both the saturating increment and floor(stamp / 2) are
    // monotonic. Victim selection consumes the non-hit prefix before the
    // probation variant stably merges the surviving non-hits with the hits.
    // This prevents a still-old hit from being selected as a victim. The legacy
    // operator keeps the faster partition-only path because all hits become 0.
    // Unique partition keys make the result independent of Sort's tie behavior:
    // nonHit * C + (C - 1 - oldIndex), sorted descending.
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
        AscendC::LocalTensor<int32_t> hitOrNonHit =
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
        AscendC::Gather(hitOrNonHit, positionMask, gatherOffsets.ReinterpretCast<uint32_t>(),
                        static_cast<uint32_t>(0), kCacheCapacity);
        AscendC::PipeBarrier<PIPE_V>();

        if (halveHitStamps != 0) {
            // hit stamp = floor(incremented stamp / 2). ShiftRight is exact for
            // the non-negative int32 stamp domain and avoids float precision
            // loss when callers choose stamp_max above 2^24.
            AscendC::ShiftRight<int32_t>(gatherOffsets, lruStamps, 1, kCacheCapacity);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(gatherOffsets, lruStamps, gatherOffsets, kCacheCapacity);
            AscendC::Mul(gatherOffsets, gatherOffsets, hitOrNonHit, kCacheCapacity);
            AscendC::Sub(lruStamps, lruStamps, gatherOffsets, kCacheCapacity);
        } else {
            AscendC::Muls(gatherOffsets, hitOrNonHit, static_cast<int32_t>(-1), kCacheCapacity);
            AscendC::Adds(gatherOffsets, gatherOffsets, static_cast<int32_t>(1), kCacheCapacity);
            AscendC::Mul(lruStamps, lruStamps, gatherOffsets, kCacheCapacity);
        }

        // Reuse the gathered hit vector as nonHit = 1 - hit.
        AscendC::Muls(hitOrNonHit, hitOrNonHit, static_cast<int32_t>(-1), kCacheCapacity);
        AscendC::Adds(hitOrNonHit, hitOrNonHit, static_cast<int32_t>(1), kCacheCapacity);

        // Build exact, unique stable-partition keys in [0, 8191].
        AscendC::Muls(hitOrNonHit, hitOrNonHit, static_cast<int32_t>(kCacheCapacity), kCacheCapacity);
        AscendC::CreateVecIndex(recordIndex, static_cast<int32_t>(0), kCacheCapacity);
        AscendC::Muls(gatherOffsets, recordIndex, static_cast<int32_t>(-1), kCacheCapacity);
        AscendC::Adds(gatherOffsets, gatherOffsets,
                      static_cast<int32_t>(kCacheCapacity - 1), kCacheCapacity);
        AscendC::Add(hitOrNonHit, hitOrNonHit, gatherOffsets, kCacheCapacity);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(recordValue, hitOrNonHit, AscendC::RoundMode::CAST_NONE, kCacheCapacity);
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

        if (halveHitStamps != 0) {
            // Locate the partition boundary from the exact sort keys. Keys for
            // non-hits are >= C and keys for hits are < C.
            SyncVectorToScalar();
            uint32_t low = 0;
            uint32_t high = kCacheCapacity;
            while (low < high) {
                const uint32_t mid = low + ((high - low) >> 1);
                if (recordValue.GetValue(mid) >= static_cast<float>(kCacheCapacity)) {
                    low = mid + 1;
                } else {
                    high = mid;
                }
            }
            nonHitCount = low;
            SyncScalarToVector();

            // BuildVictimPlan reuses the key/index arena. Preserve the original
            // positions in the dead physical-hit-mask region for the stable
            // equal-age tie break performed after victim selection.
            AscendC::Adds(positionMask, recordIndex, static_cast<int32_t>(0), kCacheCapacity);
            AscendC::PipeBarrier<PIPE_V>();
        }
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

    __aicore__ inline void WriteLruState(uint32_t requestRow, uint32_t missCount)
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

        if (halveHitStamps != 0 && nonHitCount < kCacheCapacity) {
            AscendC::LocalTensor<int32_t> stableIndices =
                workBuf.GetWithOffset<int32_t>(kCacheCapacity, kPositionMaskOffset);
            const uint32_t survivorCount = kCacheCapacity - missCount;

            // Victims are selected only from [0, missCount) of the non-hit
            // group. Merge the surviving non-hits [missCount, nonHitCount) and
            // all hits [nonHitCount, C) by descending stamp. Equal ages retain
            // the original persistent order.
            SyncVectorToScalar();
            uint32_t nonHitPos = missCount;
            uint32_t hitPos = nonHitCount;
            for (uint32_t outputPos = 0; outputPos < survivorCount; ++outputPos) {
                bool takeNonHit = false;
                if (hitPos >= kCacheCapacity) {
                    takeNonHit = true;
                } else if (nonHitPos < nonHitCount) {
                    const int32_t nonHitStamp = sortedStamps.GetValue(nonHitPos);
                    const int32_t hitStamp = sortedStamps.GetValue(hitPos);
                    takeNonHit = nonHitStamp > hitStamp ||
                                 (nonHitStamp == hitStamp &&
                                  stableIndices.GetValue(nonHitPos) < stableIndices.GetValue(hitPos));
                }
                const uint32_t sourcePos = takeNonHit ? nonHitPos++ : hitPos++;
                gatherOffsets.SetValue(outputPos,
                                       static_cast<int32_t>(sourcePos * kBytesPerInt));
            }
            SyncScalarToVector();
            AscendC::Gather(rotatedSlots, sortedSlots, gatherOffsets.ReinterpretCast<uint32_t>(),
                            static_cast<uint32_t>(0), survivorCount);
            AscendC::Gather(rotatedStamps, sortedStamps, gatherOffsets.ReinterpretCast<uint32_t>(),
                            static_cast<uint32_t>(0), survivorCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(sortedSlots[missCount], rotatedSlots, static_cast<int32_t>(0), survivorCount);
            AscendC::Adds(sortedStamps[missCount], rotatedStamps, static_cast<int32_t>(0), survivorCount);
            AscendC::PipeBarrier<PIPE_V>();
        }

        uint32_t insertionIndex = kCacheCapacity - missCount;
        if (missCount > 0) {
            if (probationAge > 0) {
                // The non-victim suffix is already sorted by descending age.
                // Find the first entry younger than probationAge so the new
                // fills remain probationary instead of becoming MRU. Existing
                // entries with the same age stay before the new fills.
                SyncVectorToScalar();
                uint32_t low = missCount;
                uint32_t high = kCacheCapacity;
                while (low < high) {
                    const uint32_t mid = low + ((high - low) >> 1);
                    if (sortedStamps.GetValue(mid) >= static_cast<int32_t>(probationAge)) {
                        low = mid + 1;
                    } else {
                        high = mid;
                    }
                }
                insertionIndex = low - missCount;
            }
            AscendC::Duplicate(sortedStamps, static_cast<int32_t>(probationAge), missCount);
            AscendC::PipeBarrier<PIPE_V>();
        }

        // Build a stable insertion permutation over the original sorted pair
        // array. Sources [missCount, ...] are the surviving entries and
        // sources [0, missCount) are the victims initialized at probationAge.
        // For probationAge == 0, insertionIndex is the suffix length and this
        // reduces to the original rotate-left-by-missCount behavior.
        AscendC::CreateVecIndex(gatherOffsets, static_cast<int32_t>(0), kCacheCapacity);

        // before = (i < insertionIndex); src += before * missCount.
        AscendC::Muls(wrapFlags, gatherOffsets, static_cast<int32_t>(-1), kCacheCapacity);
        AscendC::Adds(wrapFlags, wrapFlags, static_cast<int32_t>(insertionIndex), kCacheCapacity);
        AscendC::Maxs(wrapFlags, wrapFlags, static_cast<int32_t>(0), kCacheCapacity);
        AscendC::Mins(wrapFlags, wrapFlags, static_cast<int32_t>(1), kCacheCapacity);
        AscendC::Muls(wrapFlags, wrapFlags, static_cast<int32_t>(missCount), kCacheCapacity);
        AscendC::Add(gatherOffsets, gatherOffsets, wrapFlags, kCacheCapacity);

        // inserted = (insertionIndex <= i < insertionIndex + missCount);
        // src -= inserted * insertionIndex. rotatedSlots is dead until Gather,
        // so it serves as a second predicate scratch vector here.
        AscendC::CreateVecIndex(wrapFlags, static_cast<int32_t>(1) - static_cast<int32_t>(insertionIndex),
                                kCacheCapacity);
        AscendC::Maxs(wrapFlags, wrapFlags, static_cast<int32_t>(0), kCacheCapacity);
        AscendC::Mins(wrapFlags, wrapFlags, static_cast<int32_t>(1), kCacheCapacity);
        AscendC::CreateVecIndex(rotatedSlots,
                                static_cast<int32_t>(1) - static_cast<int32_t>(insertionIndex + missCount),
                                kCacheCapacity);
        AscendC::Maxs(rotatedSlots, rotatedSlots, static_cast<int32_t>(0), kCacheCapacity);
        AscendC::Mins(rotatedSlots, rotatedSlots, static_cast<int32_t>(1), kCacheCapacity);
        AscendC::Sub(wrapFlags, wrapFlags, rotatedSlots, kCacheCapacity);
        AscendC::Muls(wrapFlags, wrapFlags, static_cast<int32_t>(insertionIndex), kCacheCapacity);
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
    uint32_t probationAge = 0;
    uint32_t halveHitStamps = 0;
    uint32_t nonHitCount = kCacheCapacity;
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
                max_context_len, stamp_max, static_cast<uint32_t>(0), static_cast<uint32_t>(0),
                usable_ub_bytes, &pipe);
    kernel.Process();
}

extern "C" __global__ __aicore__ void fused_timestamp_lru_metadata_update_with_probation(
    GM_ADDR req_indices, GM_ADDR topk_indices, GM_ADDR device_token_pos,
    GM_ADDR hit_position_mask, GM_ADDR device_lru_slots, GM_ADDR device_lru_slot_stamps,
    GM_ADDR victim_slots, GM_ADDR miss_counts, uint32_t batch_size, uint32_t request_rows,
    uint32_t max_context_len, uint32_t stamp_max, uint32_t probation_age,
    uint32_t usable_ub_bytes)
{
    AscendC::TPipe pipe;
    KernelFusedTimestampLruMetadataUpdate kernel;
    kernel.Init(req_indices, topk_indices, device_token_pos, hit_position_mask, device_lru_slots,
                device_lru_slot_stamps, victim_slots, miss_counts, batch_size, request_rows,
                max_context_len, stamp_max, probation_age, static_cast<uint32_t>(1),
                usable_ub_bytes, &pipe);
    kernel.Process();
}
