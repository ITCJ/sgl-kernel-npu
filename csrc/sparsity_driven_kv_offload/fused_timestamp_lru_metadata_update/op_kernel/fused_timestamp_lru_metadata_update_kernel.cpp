// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// A2/A3 AIV implementation of timestamp LRU selection and metadata update.
// The kernel intentionally uses no scatter instruction. Dense transforms,
// sorting, compaction, prefix scan, and indexed reads are SIMD operations;
// only the final sparse 4-byte GM writes use DataCopyPad.

#include "kernel_operator.h"

namespace {

constexpr uint32_t kTopk = 2048;
constexpr uint32_t kCacheCapacity = 4096;
constexpr uint32_t kRecordCount = kCacheCapacity + kTopk;
constexpr uint32_t kBytesPerInt = sizeof(int32_t);
constexpr uint32_t kSortPairElements = 2;

// Stage A (6144-record hit/base sort) UB layout.
constexpr uint32_t kRecordValueOffset = 0;
constexpr uint32_t kRecordIndexOffset = kRecordValueOffset + kRecordCount * kBytesPerInt;
constexpr uint32_t kRecordSortTmpOffset = kRecordIndexOffset + kRecordCount * kBytesPerInt;
constexpr uint32_t kRecordSortOutOffset = kRecordSortTmpOffset + kSortPairElements * kRecordCount * kBytesPerInt;
constexpr uint32_t kLruSlotsOffset = kRecordSortOutOffset + kSortPairElements * kRecordCount * kBytesPerInt;
constexpr uint32_t kLruStampsOffset = kLruSlotsOffset + kCacheCapacity * kBytesPerInt;
constexpr uint32_t kTopkPosOffset = kLruStampsOffset + kCacheCapacity * kBytesPerInt;
constexpr uint32_t kWorkUbBytes = kTopkPosOffset + kTopk * kBytesPerInt;

// Stage B/C aliases. They are valid only after Stage A has consumed the
// corresponding record-sort storage.
constexpr uint32_t kStampSortValueOffset = 0;
constexpr uint32_t kStampSortIndexOffset = kCacheCapacity * kBytesPerInt;
constexpr uint32_t kStampSortTmpOffset = 2 * kCacheCapacity * kBytesPerInt;
constexpr uint32_t kStampSortOutOffset = 4 * kCacheCapacity * kBytesPerInt;
constexpr uint32_t kSortedSlotsOffset = 6 * kCacheCapacity * kBytesPerInt;
constexpr uint32_t kSortedStampsOffset = 7 * kCacheCapacity * kBytesPerInt;

constexpr uint32_t kTopkTokenOffset = 0;
constexpr uint32_t kTopkDevicePosOffset = kTopk * kBytesPerInt;
constexpr uint32_t kMissFlagOffset = 2 * kTopk * kBytesPerInt;
constexpr uint32_t kScanScratchOffset = 3 * kTopk * kBytesPerInt;
constexpr uint32_t kVictimOffset = 4 * kTopk * kBytesPerInt;
constexpr uint32_t kGatherOffsetOffset = 5 * kTopk * kBytesPerInt;
constexpr uint32_t kVectorScratchOffset = 6 * kTopk * kBytesPerInt;
constexpr uint32_t kMaskAndScalarOffset = kTopkPosOffset;

static_assert(kWorkUbBytes == 188416, "unexpected UB layout size");

__aicore__ inline void SyncMte2ToVector()
{
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
}

__aicore__ inline void SyncVectorToMte3()
{
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
}

__aicore__ inline void SyncVectorToScalar()
{
    AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
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

    __aicore__ inline void Init(GM_ADDR slotMap, GM_ADDR reqIndices, GM_ADDR topkIndices,
                                GM_ADDR deviceTokenPos, GM_ADDR deviceLruSlots, GM_ADDR deviceLruSlotStamps,
                                GM_ADDR deviceSlotTokens, GM_ADDR victimSlots, uint32_t batchSize,
                                uint32_t requestRows, uint32_t slotMapRows, uint32_t slotMapWidth,
                                uint32_t maxContextLen, uint32_t stampMax, uint32_t usableUbBytes,
                                AscendC::TPipe *pipe)
    {
        this->batchSize = batchSize;
        this->requestRows = requestRows;
        this->slotMapRows = slotMapRows;
        this->slotMapWidth = slotMapWidth;
        this->maxContextLen = maxContextLen;
        this->stampMax = stampMax;

        slotMapGm.SetGlobalBuffer((__gm__ int32_t *)slotMap,
                                  static_cast<uint64_t>(slotMapRows) * slotMapWidth);
        reqIndicesGm.SetGlobalBuffer((__gm__ int32_t *)reqIndices, batchSize);
        topkIndicesGm.SetGlobalBuffer((__gm__ int32_t *)topkIndices,
                                     static_cast<uint64_t>(batchSize) * kTopk);
        deviceTokenPosGm.SetGlobalBuffer((__gm__ int32_t *)deviceTokenPos,
                                        static_cast<uint64_t>(batchSize) * kTopk);
        deviceLruSlotsGm.SetGlobalBuffer((__gm__ int32_t *)deviceLruSlots,
                                        static_cast<uint64_t>(requestRows) * kCacheCapacity);
        deviceLruSlotStampsGm.SetGlobalBuffer((__gm__ int32_t *)deviceLruSlotStamps,
                                             static_cast<uint64_t>(requestRows) * kCacheCapacity);
        deviceSlotTokensGm.SetGlobalBuffer((__gm__ int32_t *)deviceSlotTokens,
                                          static_cast<uint64_t>(requestRows) * kCacheCapacity);
        victimSlotsGm.SetGlobalBuffer((__gm__ int32_t *)victimSlots,
                                     static_cast<uint64_t>(batchSize) * kTopk);

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
        // Request row 0 is reserved by the caller for graph padding. Sentinel
        // and out-of-range rows also return all -1 victims without mutation.
        if (reqId <= 0 || static_cast<uint32_t>(reqId) >= requestRows ||
            static_cast<uint32_t>(reqId) >= slotMapRows) {
            WriteInvalidVictims(batchIdx);
            return;
        }

        const uint32_t requestRow = static_cast<uint32_t>(reqId);
        BuildUpdatedSlotStampPairs(batchIdx, requestRow);
        SortSlotsByStamp();
        const uint32_t missCount = BuildVictimPlan(batchIdx);
        UpdateSparseMetadata(batchIdx, requestRow, missCount);
        WriteRotatedLruState(requestRow, missCount);
    }

    __aicore__ inline void WriteInvalidVictims(uint32_t batchIdx)
    {
        AscendC::LocalTensor<int32_t> invalid = workBuf.GetWithOffset<int32_t>(kTopk, 0);
        AscendC::Duplicate(invalid, static_cast<int32_t>(-1), kTopk);
        SyncVectorToMte3();
        CopyRowOut(victimSlotsGm[batchIdx * kTopk], invalid, kTopk);
    }

    // Sort C base records and K hit records by (physical_slot + tie), with a
    // base record key of slot+0.25 and a hit key of slot. Descending sort puts
    // the unique base first in every physical-slot group. The following record
    // has the same rounded slot iff that slot was hit, including duplicate hits.
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
        AscendC::LocalTensor<int32_t> devicePos =
            workBuf.GetWithOffset<int32_t>(kTopk, kTopkPosOffset);

        CopyRowIn(lruSlots, deviceLruSlotsGm[requestRow * kCacheCapacity], kCacheCapacity);
        CopyRowIn(lruStamps, deviceLruSlotStampsGm[requestRow * kCacheCapacity], kCacheCapacity);
        CopyRowIn(devicePos, deviceTokenPosGm[batchIdx * kTopk], kTopk);
        SyncMte2ToVector();

        // Saturating age increment: min(stamp, stampMax - 1) + 1.
        AscendC::Mins(lruStamps, lruStamps, static_cast<int32_t>(stampMax - 1), kCacheCapacity);
        AscendC::Adds(lruStamps, lruStamps, static_cast<int32_t>(1), kCacheCapacity);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Cast(recordValue, lruSlots, AscendC::RoundMode::CAST_NONE, kCacheCapacity);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(recordValue, recordValue, 0.25f, kCacheCapacity);
        AscendC::Cast(recordValue[kCacheCapacity], devicePos, AscendC::RoundMode::CAST_NONE, kTopk);
        AscendC::CreateVecIndex(recordIndex, static_cast<int32_t>(0), kRecordCount);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Sort<float, true>(sortOut, recordValue, recordIndex.ReinterpretCast<uint32_t>(), sortTmp,
                                   kRecordCount / 32);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Extract(recordValue, recordIndex.ReinterpretCast<uint32_t>(), sortOut, kRecordCount / 32);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::LocalTensor<int32_t> physical =
            workBuf.GetWithOffset<int32_t>(kRecordCount, kRecordSortOutOffset);
        AscendC::LocalTensor<int32_t> nextPhysical =
            workBuf.GetWithOffset<int32_t>(kRecordCount,
                                           kRecordSortOutOffset + kRecordCount * kBytesPerInt);
        AscendC::LocalTensor<int32_t> gatherOffsets =
            workBuf.GetWithOffset<int32_t>(kRecordCount, kRecordSortTmpOffset);
        AscendC::LocalTensor<int32_t> gatheredStamps =
            workBuf.GetWithOffset<int32_t>(kRecordCount,
                                           kRecordSortTmpOffset + kRecordCount * kBytesPerInt);

        // RINT maps slot+0.25 and slot to the same exact physical slot.
        AscendC::Cast(physical, recordValue, AscendC::RoundMode::CAST_RINT, kRecordCount);
        AscendC::CreateVecIndex(gatherOffsets, static_cast<int32_t>(1), kRecordCount);
        AscendC::Mins(gatherOffsets, gatherOffsets, static_cast<int32_t>(kRecordCount - 1), kRecordCount);
        AscendC::Muls(gatherOffsets, gatherOffsets, static_cast<int32_t>(kBytesPerInt), kRecordCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Gather(nextPhysical, physical, gatherOffsets.ReinterpretCast<uint32_t>(),
                        static_cast<uint32_t>(0), kRecordCount);
        AscendC::PipeBarrier<PIPE_V>();

        // nextPhysical becomes 1 for an unhit base and 0 for a hit base.
        AscendC::Sub(nextPhysical, nextPhysical, physical, kRecordCount);
        AscendC::Abs(nextPhysical, nextPhysical, kRecordCount);
        AscendC::Mins(nextPhysical, nextPhysical, static_cast<int32_t>(1), kRecordCount);
        AscendC::PipeBarrier<PIPE_V>();

        // Gather incremented stamps by the base record's original LRU index.
        AscendC::Mins(gatherOffsets, recordIndex,
                      static_cast<int32_t>(kCacheCapacity - 1), kRecordCount);
        AscendC::Muls(gatherOffsets, gatherOffsets, static_cast<int32_t>(kBytesPerInt), kRecordCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Gather(gatheredStamps, lruStamps, gatherOffsets.ReinterpretCast<uint32_t>(),
                        static_cast<uint32_t>(0), kRecordCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(gatheredStamps, gatheredStamps, nextPhysical, kRecordCount);
        AscendC::PipeBarrier<PIPE_V>();

        // Compact exactly the C base records. This is the scatter-free hit
        // update: a base whose next group member is a hit receives stamp zero.
        AscendC::LocalTensor<uint32_t> baseMask =
            workBuf.GetWithOffset<uint32_t>((kRecordCount + 31) / 32, kMaskAndScalarOffset);
        AscendC::CompareScalar(baseMask.ReinterpretCast<uint8_t>(), recordIndex,
                               static_cast<int32_t>(kCacheCapacity), AscendC::CMPMODE::LT, kRecordCount);
        AscendC::PipeBarrier<PIPE_V>();

        uint64_t compactSlotCount = 0;
        uint64_t compactStampCount = 0;
        AscendC::GatherMask(lruSlots, physical, baseMask, true, kRecordCount, {1, 1, 0, 0}, compactSlotCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::GatherMask(lruStamps, gatheredStamps, baseMask, true, kRecordCount, {1, 1, 0, 0},
                            compactStampCount);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void SortSlotsByStamp()
    {
        AscendC::LocalTensor<float> stampValues =
            workBuf.GetWithOffset<float>(kCacheCapacity, kStampSortValueOffset);
        AscendC::LocalTensor<int32_t> stampIndices =
            workBuf.GetWithOffset<int32_t>(kCacheCapacity, kStampSortIndexOffset);
        AscendC::LocalTensor<float> sortTmp =
            workBuf.GetWithOffset<float>(2 * kCacheCapacity, kStampSortTmpOffset);
        AscendC::LocalTensor<float> sortOut =
            workBuf.GetWithOffset<float>(2 * kCacheCapacity, kStampSortOutOffset);
        AscendC::LocalTensor<int32_t> compactSlots =
            workBuf.GetWithOffset<int32_t>(kCacheCapacity, kLruSlotsOffset);
        AscendC::LocalTensor<int32_t> compactStamps =
            workBuf.GetWithOffset<int32_t>(kCacheCapacity, kLruStampsOffset);

        AscendC::Cast(stampValues, compactStamps, AscendC::RoundMode::CAST_NONE, kCacheCapacity);
        AscendC::CreateVecIndex(stampIndices, static_cast<int32_t>(0), kCacheCapacity);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sort<float, true>(sortOut, stampValues, stampIndices.ReinterpretCast<uint32_t>(), sortTmp,
                                   kCacheCapacity / 32);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Extract(stampValues, stampIndices.ReinterpretCast<uint32_t>(), sortOut, kCacheCapacity / 32);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::LocalTensor<int32_t> sortedSlots =
            workBuf.GetWithOffset<int32_t>(kCacheCapacity, kSortedSlotsOffset);
        AscendC::LocalTensor<int32_t> sortedStamps =
            workBuf.GetWithOffset<int32_t>(kCacheCapacity, kSortedStampsOffset);
        AscendC::Muls(stampIndices, stampIndices, static_cast<int32_t>(kBytesPerInt), kCacheCapacity);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Gather(sortedSlots, compactSlots, stampIndices.ReinterpretCast<uint32_t>(),
                        static_cast<uint32_t>(0), kCacheCapacity);
        AscendC::Gather(sortedStamps, compactStamps, stampIndices.ReinterpretCast<uint32_t>(),
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

        // Keep the recovered miss vector for the scalar-address DataCopyPad
        // update loop.
        AscendC::Adds(missFlag, scanScratch, static_cast<int32_t>(0), kTopk);
        SyncVectorToMte3();
        CopyRowOut(victimSlotsGm[batchIdx * kTopk], victims, kTopk);
        return missCount;
    }

    __aicore__ inline void UpdateSparseMetadata(uint32_t batchIdx, uint32_t requestRow, uint32_t missCount)
    {
        if (missCount == 0) {
            return;
        }

        AscendC::LocalTensor<int32_t> topkTokens =
            workBuf.GetWithOffset<int32_t>(kTopk, kTopkTokenOffset);
        AscendC::LocalTensor<int32_t> missFlag =
            workBuf.GetWithOffset<int32_t>(kTopk, kMissFlagOffset);
        AscendC::LocalTensor<int32_t> victims =
            workBuf.GetWithOffset<int32_t>(kTopk, kVictimOffset);
        AscendC::LocalTensor<int32_t> slotTokens =
            workBuf.GetWithOffset<int32_t>(kCacheCapacity, kLruSlotsOffset);
        AscendC::LocalTensor<int32_t> minusOne =
            workBuf.GetWithOffset<int32_t>(8, kMaskAndScalarOffset);

        CopyRowIn(slotTokens, deviceSlotTokensGm[requestRow * kCacheCapacity], kCacheCapacity);
        AscendC::Duplicate(minusOne, static_cast<int32_t>(-1), 8);
        SyncMte2ToVector();
        SyncVectorToScalar();
        SyncVectorToMte3();

        AscendC::DataCopyExtParams oneIntParams{1, sizeof(int32_t), 0, 0, 0};
        for (uint32_t i = 0; i < kTopk; ++i) {
            if (missFlag.GetValue(i) == 0) {
                continue;
            }
            const int32_t victim = victims.GetValue(i);
            const int32_t newToken = topkTokens.GetValue(i);
            if (victim < 0 || static_cast<uint32_t>(victim) >= kCacheCapacity ||
                newToken < 0 || static_cast<uint32_t>(newToken) >= maxContextLen) {
                continue;
            }

            const int32_t oldToken = slotTokens.GetValue(static_cast<uint32_t>(victim));
            if (oldToken >= 0 && static_cast<uint32_t>(oldToken) < maxContextLen) {
                const uint32_t oldMapOffset = requestRow * slotMapWidth + static_cast<uint32_t>(oldToken);
                AscendC::DataCopyPad(slotMapGm[oldMapOffset], minusOne, oneIntParams);
            }

            const uint32_t slotTokenOffset = requestRow * kCacheCapacity + static_cast<uint32_t>(victim);
            AscendC::DataCopyPad(deviceSlotTokensGm[slotTokenOffset], topkTokens[i], oneIntParams);
            const uint32_t newMapOffset = requestRow * slotMapWidth + static_cast<uint32_t>(newToken);
            AscendC::DataCopyPad(slotMapGm[newMapOffset], victims[i], oneIntParams);
        }
    }

    __aicore__ inline void WriteRotatedLruState(uint32_t requestRow, uint32_t missCount)
    {
        AscendC::LocalTensor<int32_t> sortedSlots =
            workBuf.GetWithOffset<int32_t>(kCacheCapacity, kSortedSlotsOffset);
        AscendC::LocalTensor<int32_t> sortedStamps =
            workBuf.GetWithOffset<int32_t>(kCacheCapacity, kSortedStampsOffset);

        // Newly occupied victims become MRU (stamp zero). Rotating them to the
        // tail keeps the persistent pair array in descending stamp order.
        if (missCount > 0) {
            AscendC::Duplicate(sortedStamps, static_cast<int32_t>(0), missCount);
            AscendC::PipeBarrier<PIPE_V>();
        }
        SyncVectorToMte3();

        const uint32_t remaining = kCacheCapacity - missCount;
        const uint32_t rowOffset = requestRow * kCacheCapacity;
        CopyRowOut(deviceLruSlotsGm[rowOffset], sortedSlots[missCount], remaining);
        CopyRowOut(deviceLruSlotStampsGm[rowOffset], sortedStamps[missCount], remaining);
        CopyRowOut(deviceLruSlotsGm[rowOffset + remaining], sortedSlots, missCount);
        CopyRowOut(deviceLruSlotStampsGm[rowOffset + remaining], sortedStamps, missCount);
    }

private:
    AscendC::TBuf<AscendC::TPosition::VECCALC> workBuf;
    AscendC::GlobalTensor<int32_t> slotMapGm;
    AscendC::GlobalTensor<int32_t> reqIndicesGm;
    AscendC::GlobalTensor<int32_t> topkIndicesGm;
    AscendC::GlobalTensor<int32_t> deviceTokenPosGm;
    AscendC::GlobalTensor<int32_t> deviceLruSlotsGm;
    AscendC::GlobalTensor<int32_t> deviceLruSlotStampsGm;
    AscendC::GlobalTensor<int32_t> deviceSlotTokensGm;
    AscendC::GlobalTensor<int32_t> victimSlotsGm;

    uint32_t batchSize = 0;
    uint32_t requestRows = 0;
    uint32_t slotMapRows = 0;
    uint32_t slotMapWidth = 0;
    uint32_t maxContextLen = 0;
    uint32_t stampMax = 0;
};

}  // namespace

extern "C" __global__ __aicore__ void fused_timestamp_lru_metadata_update(
    GM_ADDR slot_map, GM_ADDR req_indices, GM_ADDR topk_indices, GM_ADDR device_token_pos,
    GM_ADDR device_lru_slots, GM_ADDR device_lru_slot_stamps, GM_ADDR device_slot_tokens,
    GM_ADDR victim_slots, uint32_t batch_size, uint32_t request_rows, uint32_t slot_map_rows,
    uint32_t slot_map_width, uint32_t max_context_len, uint32_t stamp_max, uint32_t usable_ub_bytes)
{
    AscendC::TPipe pipe;
    KernelFusedTimestampLruMetadataUpdate kernel;
    kernel.Init(slot_map, req_indices, topk_indices, device_token_pos, device_lru_slots,
                device_lru_slot_stamps, device_slot_tokens, victim_slots, batch_size, request_rows,
                slot_map_rows, slot_map_width, max_context_len, stamp_max, usable_ub_bytes, &pipe);
    kernel.Process();
}
