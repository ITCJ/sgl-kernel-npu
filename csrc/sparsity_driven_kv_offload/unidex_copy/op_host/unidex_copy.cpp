// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "defines.h"
#include "torch_helper.h"

#include "aclrtlaunch_unidex_copy.h"
#include "aclrtlaunch_uindex_copy_optimized.h"

#include <cstdint>
#include <limits>

namespace sglang {
namespace npu_kernel {

namespace {

constexpr int64_t kMaxBlockBytes = 32 * 1024;
constexpr uint64_t kUint32Max = std::numeric_limits<uint32_t>::max();

void CheckNpuTensor(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.device().type() == at::DeviceType::PrivateUse1, name, " must be on an NPU device");
}

void CheckSameDevice(const at::Tensor &tensor, const at::Tensor &reference, const char *name)
{
    TORCH_CHECK(tensor.device() == reference.device(), name, " must be on the same device as the reference tensor");
}

void CheckFitsUint32(int64_t value, const char *name)
{
    TORCH_CHECK(value >= 0 && static_cast<uint64_t>(value) <= kUint32Max, name, " exceeds uint32 range: ", value);
}

struct CopyLaunchParams {
    void *srcAddr;
    void *dstAddr;
    uint32_t srcRows;
    uint32_t dstRows;
    uint32_t blockBytes;
    uint32_t maxCopy;
    uint32_t blockDim;
};

CopyLaunchParams PrepareCopyLaunch(const at::Tensor &src, at::Tensor &dst, const at::Tensor &srcIndex,
                                   const at::Tensor &dstIndex, const at::Tensor &validMask, int64_t srcRows,
                                   int64_t dstRows, int64_t blockBytes, int64_t maxCopy, int64_t blockDim,
                                   c10::optional<int64_t> srcPtr, c10::optional<int64_t> dstPtr)
{
    const bool useRawSrc = srcPtr.has_value();
    const bool useRawDst = dstPtr.has_value();

    CheckNpuTensor(srcIndex, "src_index");
    CheckNpuTensor(dstIndex, "dst_index");
    CheckNpuTensor(validMask, "valid_mask");
    CheckSameDevice(dstIndex, srcIndex, "dst_index");
    CheckSameDevice(validMask, srcIndex, "valid_mask");
    if (!useRawSrc) {
        CheckNpuTensor(src, "src");
        CheckSameDevice(src, srcIndex, "src");
    }
    if (!useRawDst) {
        CheckNpuTensor(dst, "dst");
        CheckSameDevice(dst, srcIndex, "dst");
    }

    TORCH_CHECK(src.is_contiguous(), "src must be contiguous");
    TORCH_CHECK(dst.is_contiguous(), "dst must be contiguous");
    TORCH_CHECK(srcIndex.is_contiguous(), "src_index must be contiguous");
    TORCH_CHECK(dstIndex.is_contiguous(), "dst_index must be contiguous");
    TORCH_CHECK(validMask.is_contiguous(), "valid_mask must be contiguous");
    TORCH_CHECK(src.scalar_type() == dst.scalar_type(), "src and dst must have the same dtype, got ", src.scalar_type(),
                " and ", dst.scalar_type());
    TORCH_CHECK(srcIndex.scalar_type() == at::kLong, "src_index must be int64, got ", srcIndex.scalar_type());
    TORCH_CHECK(dstIndex.scalar_type() == at::kLong, "dst_index must be int64, got ", dstIndex.scalar_type());
    TORCH_CHECK(validMask.scalar_type() == at::kBool || validMask.scalar_type() == at::kByte,
                "valid_mask must be bool or uint8, got ", validMask.scalar_type());
    TORCH_CHECK(srcIndex.dim() == 1, "src_index must be 1-D, got ", srcIndex.dim());
    TORCH_CHECK(dstIndex.dim() == 1, "dst_index must be 1-D, got ", dstIndex.dim());
    TORCH_CHECK(validMask.dim() == 1, "valid_mask must be 1-D, got ", validMask.dim());

    TORCH_CHECK(!useRawSrc || *srcPtr > 0, "src_ptr must be a non-zero address");
    TORCH_CHECK(!useRawDst || *dstPtr > 0, "dst_ptr must be a non-zero address");
    TORCH_CHECK(srcRows > 0, "src_rows must be positive, got ", srcRows);
    TORCH_CHECK(dstRows > 0, "dst_rows must be positive, got ", dstRows);
    TORCH_CHECK(blockBytes > 0, "block_bytes must be positive, got ", blockBytes);
    TORCH_CHECK(blockBytes <= kMaxBlockBytes, "block_bytes exceeds the supported 32 KiB limit: ", blockBytes);
    TORCH_CHECK(maxCopy >= 0, "max_copy must be non-negative, got ", maxCopy);
    TORCH_CHECK(blockDim > 0, "block_dim must be positive, got ", blockDim);

    CheckFitsUint32(srcRows, "src_rows");
    CheckFitsUint32(dstRows, "dst_rows");
    CheckFitsUint32(blockBytes, "block_bytes");
    CheckFitsUint32(maxCopy, "max_copy");
    CheckFitsUint32(blockDim, "block_dim");

    TORCH_CHECK(srcIndex.numel() >= maxCopy, "src_index has ", srcIndex.numel(),
                " elements, fewer than max_copy=", maxCopy);
    TORCH_CHECK(dstIndex.numel() >= maxCopy, "dst_index has ", dstIndex.numel(),
                " elements, fewer than max_copy=", maxCopy);
    TORCH_CHECK(validMask.numel() >= maxCopy, "valid_mask has ", validMask.numel(),
                " elements, fewer than max_copy=", maxCopy);

    const uint64_t srcRequiredBytes = static_cast<uint64_t>(srcRows) * static_cast<uint64_t>(blockBytes);
    const uint64_t dstRequiredBytes = static_cast<uint64_t>(dstRows) * static_cast<uint64_t>(blockBytes);
    TORCH_CHECK(srcRequiredBytes <= kUint32Max, "src_rows * block_bytes exceeds the kernel uint32 address range");
    TORCH_CHECK(dstRequiredBytes <= kUint32Max, "dst_rows * block_bytes exceeds the kernel uint32 address range");
    if (!useRawSrc) {
        const uint64_t srcAvailableBytes =
            static_cast<uint64_t>(src.numel()) * static_cast<uint64_t>(src.element_size());
        TORCH_CHECK(srcRequiredBytes <= srcAvailableBytes, "src storage is too small: need ", srcRequiredBytes,
                    " bytes, got ", srcAvailableBytes);
    }
    if (!useRawDst) {
        const uint64_t dstAvailableBytes =
            static_cast<uint64_t>(dst.numel()) * static_cast<uint64_t>(dst.element_size());
        TORCH_CHECK(dstRequiredBytes <= dstAvailableBytes, "dst storage is too small: need ", dstRequiredBytes,
                    " bytes, got ", dstAvailableBytes);
    }

    if (maxCopy > 0) {
        auto npuStream = c10_npu::getCurrentNPUStream();
        if (!useRawSrc) {
            src.record_stream(npuStream);
        }
        if (!useRawDst) {
            dst.record_stream(npuStream);
        }
        srcIndex.record_stream(npuStream);
        dstIndex.record_stream(npuStream);
        validMask.record_stream(npuStream);
    }

    void *srcAddr =
        useRawSrc ? reinterpret_cast<void *>(static_cast<uintptr_t>(*srcPtr)) : const_cast<void *>(src.data_ptr());
    void *dstAddr = useRawDst ? reinterpret_cast<void *>(static_cast<uintptr_t>(*dstPtr)) : dst.data_ptr();
    return {srcAddr, dstAddr, static_cast<uint32_t>(srcRows), static_cast<uint32_t>(dstRows),
            static_cast<uint32_t>(blockBytes), static_cast<uint32_t>(maxCopy), static_cast<uint32_t>(blockDim)};
}

}  // namespace

/*
 * For every i where valid_mask[i] is true:
 *   dst[dst_index[i]] = src[src_index[i]]
 *
 * src_ptr and dst_ptr optionally override the Tensor storage addresses with
 * device-visible shared-memory addresses. Their lifetime is owned by the
 * caller and must extend through completion on the current NPU stream.
 */
HOST_API void unidex_copy(const at::Tensor &src, at::Tensor &dst, const at::Tensor &src_index,
                          const at::Tensor &dst_index, const at::Tensor &valid_mask, int64_t src_rows, int64_t dst_rows,
                          int64_t block_bytes, int64_t max_copy, int64_t block_dim, c10::optional<int64_t> src_ptr,
                          c10::optional<int64_t> dst_ptr)
{
    const CopyLaunchParams params = PrepareCopyLaunch(src, dst, src_index, dst_index, valid_mask, src_rows, dst_rows,
                                                      block_bytes, max_copy, block_dim, src_ptr, dst_ptr);
    if (params.maxCopy == 0) {
        return;
    }

    void *srcAddr = params.srcAddr;
    void *dstAddr = params.dstAddr;
    uint32_t srcRowsU32 = params.srcRows;
    uint32_t dstRowsU32 = params.dstRows;
    uint32_t blockBytesU32 = params.blockBytes;
    uint32_t maxCopyU32 = params.maxCopy;
    uint32_t blockDimU32 = params.blockDim;

    EXEC_KERNEL_CMD(unidex_copy, blockDimU32, srcAddr, dstAddr, src_index, dst_index, valid_mask, srcRowsU32,
                    dstRowsU32, blockBytesU32, maxCopyU32);
}

/*
 * Interleaved full-row variant of unidex_copy.
 *
 * Core c handles mappings c, c + block_dim, c + 2 * block_dim, ... . This
 * distributes a valid prefix across all launched AIVs without expanding the
 * mapping tensors or splitting rows into small DMA transfers.
 */
HOST_API void uindex_copy_optimized(const at::Tensor &src, at::Tensor &dst, const at::Tensor &src_index,
                                    const at::Tensor &dst_index, const at::Tensor &valid_mask, int64_t src_rows,
                                    int64_t dst_rows, int64_t block_bytes, int64_t max_copy, int64_t block_dim,
                                    c10::optional<int64_t> src_ptr, c10::optional<int64_t> dst_ptr)
{
    const CopyLaunchParams params = PrepareCopyLaunch(src, dst, src_index, dst_index, valid_mask, src_rows, dst_rows,
                                                      block_bytes, max_copy, block_dim, src_ptr, dst_ptr);
    if (params.maxCopy == 0) {
        return;
    }

    void *srcAddr = params.srcAddr;
    void *dstAddr = params.dstAddr;
    uint32_t srcRowsU32 = params.srcRows;
    uint32_t dstRowsU32 = params.dstRows;
    uint32_t blockBytesU32 = params.blockBytes;
    uint32_t maxCopyU32 = params.maxCopy;
    uint32_t blockDimU32 = params.blockDim;

    EXEC_KERNEL_CMD(uindex_copy_optimized, blockDimU32, srcAddr, dstAddr, src_index, dst_index, valid_mask, srcRowsU32,
                    dstRowsU32, blockBytesU32, maxCopyU32);
}

}  // namespace npu_kernel
}  // namespace sglang
