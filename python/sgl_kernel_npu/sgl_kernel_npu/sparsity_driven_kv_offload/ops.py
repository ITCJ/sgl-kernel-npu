import ctypes
from math import prod
from typing import Optional, Sequence, Tuple, Union

import torch


def _ctype_for_dtype(dtype: torch.dtype):
    if dtype in (torch.float16, torch.bfloat16):
        return ctypes.c_uint16
    if dtype == torch.float32:
        return ctypes.c_float
    if dtype == torch.float64:
        return ctypes.c_double
    if dtype == torch.int8:
        return ctypes.c_int8
    if dtype == torch.uint8:
        return ctypes.c_uint8
    if dtype == torch.int16:
        return ctypes.c_int16
    if dtype == torch.int32:
        return ctypes.c_int32
    if dtype == torch.int64:
        return ctypes.c_int64
    if dtype == torch.bool:
        return ctypes.c_bool
    raise TypeError(f"unsupported shm tensor dtype: {dtype}")


def create_shm_tensor(
    shape: Sequence[int],
    dtype: torch.dtype,
    device_id: int = 0,
    name: str = "",
) -> Tuple[torch.Tensor, int, int]:
    """Create host shared memory and register it to an NPU device.

    Returns ``(host_tensor, host_ptr, dev_ptr)``. ``host_tensor`` is a CPU
    tensor backed by the registered shared memory. ``dev_ptr`` is the
    device-visible address and can be passed to sparse KV kernels through
    ``src_ptr``/``dst_ptr``.
    """
    shape_tuple = tuple(int(dim) for dim in shape)
    if any(dim < 0 for dim in shape_tuple):
        raise ValueError(f"shape dimensions must be non-negative, got {shape_tuple}")

    numel = int(prod(shape_tuple))
    elem_size = torch.empty((), dtype=dtype).element_size()
    size = numel * elem_size
    if size <= 0:
        raise ValueError(f"shm tensor size must be positive, got shape={shape_tuple}")

    host_ptr, dev_ptr = torch.ops.npu.shm_allocator_create_and_register(
        size, device_id, name
    )
    buffer_type = _ctype_for_dtype(dtype) * numel
    buffer = buffer_type.from_address(host_ptr)
    tensor = torch.frombuffer(buffer, dtype=dtype).view(shape_tuple)
    if tensor.element_size() != elem_size:
        raise RuntimeError("shm tensor element size mismatch")
    tensor.zero_()
    return tensor, int(host_ptr), int(dev_ptr)


def free_shm(device_id: int = 0) -> None:
    """Free all shared-memory allocations registered by this process."""
    torch.ops.npu.shm_allocator_free_all(device_id)


def _infer_rows_and_block_bytes(
    tensor: torch.Tensor, address_ndims: int, name: str
) -> Tuple[int, int]:
    if address_ndims <= 0 or address_ndims >= tensor.dim():
        raise ValueError(
            f"{name}_address_ndims must be in [1, {tensor.dim() - 1}], "
            f"got {address_ndims}"
        )

    rows = prod(tensor.shape[:address_ndims])
    block_elements = prod(tensor.shape[address_ndims:])
    return int(rows), int(block_elements * tensor.element_size())


def unidex_copy_inplace(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_index: torch.Tensor,
    dst_index: torch.Tensor,
    valid_mask: torch.Tensor,
    src_address_ndims: int,
    dst_address_ndims: int,
    block_dim: int = 8,
    src_ptr: Optional[int] = None,
    dst_ptr: Optional[int] = None,
) -> torch.Tensor:
    """Copy selected logical rows from ``src`` into ``dst`` in place.

    ``src_ptr`` and ``dst_ptr`` may override the Tensor addresses with
    device-visible shared-memory addresses. The caller owns those allocations
    and must keep them alive until work on the current NPU stream completes.
    """
    src_rows, src_block_bytes = _infer_rows_and_block_bytes(
        src, src_address_ndims, "src"
    )
    dst_rows, dst_block_bytes = _infer_rows_and_block_bytes(
        dst, dst_address_ndims, "dst"
    )
    if src_block_bytes != dst_block_bytes:
        raise ValueError(
            "src and dst logical rows must have the same byte size, got "
            f"{src_block_bytes} and {dst_block_bytes}"
        )
    if (
        src_index.numel() != dst_index.numel()
        or src_index.numel() != valid_mask.numel()
    ):
        raise ValueError(
            "src_index, dst_index, and valid_mask must have the same length"
        )
    if src.dtype != dst.dtype:
        raise ValueError(
            f"src and dst must have the same dtype, got {src.dtype} and {dst.dtype}"
        )

    torch.ops.npu.unidex_copy(
        src,
        dst,
        src_index,
        dst_index,
        valid_mask,
        src_rows,
        dst_rows,
        src_block_bytes,
        src_index.numel(),
        block_dim,
        src_ptr,
        dst_ptr,
    )
    return dst


def slot_map_lookup(
    slot_map: torch.Tensor,
    req_indices: torch.Tensor,
    topk_indices: torch.Tensor,
    block_dim: int = 0,
    pos_mask_size: Optional[int] = None,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """Return cache-hit flags and slot positions for ``topk_indices``.

    When ``pos_mask_size`` is provided, also return an int32 mask with shape
    ``[bs, pos_mask_size]``. For every cache hit at position ``pos``, the
    kernel sets ``position_mask[b, pos] = 1``. Hit positions outside the mask
    range are ignored. Omitting ``pos_mask_size`` preserves the legacy
    two-output API and disables position-mask writes in the kernel.
    """
    token_on_device = torch.empty_like(topk_indices, dtype=torch.int32)
    device_token_pos = torch.empty_like(topk_indices, dtype=torch.int32)
    if pos_mask_size is None:
        effective_pos_mask_size = 0
        position_mask = torch.empty(
            (topk_indices.size(0), 0),
            dtype=torch.int32,
            device=topk_indices.device,
        )
    else:
        effective_pos_mask_size = int(pos_mask_size)
        if effective_pos_mask_size <= 0:
            raise ValueError(
                f"pos_mask_size must be positive when provided, got {pos_mask_size}"
            )
        if effective_pos_mask_size % 8 != 0:
            raise ValueError(
                "pos_mask_size must be a multiple of 8 for aligned atomic mask "
                f"updates, got {pos_mask_size}"
            )
        position_mask = torch.empty(
            (topk_indices.size(0), effective_pos_mask_size),
            dtype=torch.int32,
            device=topk_indices.device,
        )
    torch.ops.npu.slot_map_lookup(
        slot_map,
        req_indices,
        topk_indices,
        token_on_device,
        device_token_pos,
        position_mask,
        effective_pos_mask_size,
        block_dim,
    )
    if pos_mask_size is not None:
        return token_on_device, device_token_pos, position_mask
    return token_on_device, device_token_pos
