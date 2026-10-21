// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// K-cooperative SYCL linear weights with a scalar fallback for raw F16/BF16/F32.
#include "strata/kernels/raw_linear.hpp"

#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/core/runtime.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_raw_linear
#include "strata/kernels/device_arch_tag.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace strata::kernels::raw {
namespace {

sycl::queue& queue_for(void* stream) { return strata::core::Runtime::get().stream(stream); }

void sync_if_needed(void* stream, const sycl::event& event, const char* what) {
    if (!stream) strata::core::Runtime::get().wait(event, what);
}

void validate(const LinearView& view, const float* x, float* y, int64_t ldx, int64_t ldy, int64_t rows) {
    if (view.data == nullptr || x == nullptr || y == nullptr)
        throw std::invalid_argument("raw::linear_fp32: data, x and y are required");
    if (view.k <= 0 || view.n <= 0 || rows <= 0 || ldx < view.k || ldy < view.n)
        throw std::invalid_argument("raw::linear_fp32: invalid dimensions or leading dimensions");
    if (view.dtype != DType::F16 && view.dtype != DType::BF16 && view.dtype != DType::F32)
        throw std::invalid_argument("raw::linear_fp32: unsupported dtype");
    const uint64_t max_bytes = (uint64_t) std::numeric_limits<ptrdiff_t>::max();
    const uint64_t max_weight_count = max_bytes / (uint64_t) dtype_bytes(view.dtype);
    if ((uint64_t) view.n > std::numeric_limits<uint64_t>::max() / (uint64_t) view.k ||
        (uint64_t) view.k * (uint64_t) view.n > max_weight_count ||
        (uint64_t) rows > max_bytes / sizeof(float) / (uint64_t) ldx ||
        (uint64_t) rows > max_bytes / sizeof(float) / (uint64_t) ldy ||
        (view.bias != nullptr && (uint64_t) view.n > max_bytes / sizeof(float)))
        throw std::invalid_argument("raw::linear_fp32: caller span exceeds host address range");
}

bool cooperative_capable(const sycl::queue& queue) {
    const auto sizes = queue.get_device().get_info<sycl::info::device::sub_group_sizes>();
    return std::find(sizes.begin(), sizes.end(), size_t(32)) != sizes.end() &&
           queue.get_device().get_info<sycl::info::device::max_work_group_size>() >= 32;
}

sycl::event linear_fp32_scalar(sycl::queue& queue, const LinearView& view, const float* x, int64_t ldx, float* y,
                               int64_t ldy, int64_t rows, const std::vector<sycl::event>& deps) {
    return queue.submit([&](sycl::handler& h) {
        if (!deps.empty()) h.depends_on(deps);
        h.parallel_for(sycl::range<1>((size_t) rows * (size_t) view.n), [=](sycl::id<1> id) {
            const size_t flat = (size_t) id[0];
            const int64_t row = (int64_t) (flat / (size_t) view.n);
            const int64_t out = (int64_t) (flat % (size_t) view.n);
            const float* xr = x + row * ldx;
            float value = 0.0f;
            const size_t base = (size_t) out * (size_t) view.k;
            for (int64_t in = 0; in < view.k; ++in) {
                float weight;
                if (view.dtype == DType::F16)
                    weight = f32_from_f16(static_cast<const uint16_t*>(view.data)[base + (size_t) in]);
                else if (view.dtype == DType::BF16)
                    weight = f32_from_bf16(static_cast<const uint16_t*>(view.data)[base + (size_t) in]);
                else
                    weight = static_cast<const float*>(view.data)[base + (size_t) in];
                value += xr[in] * weight;
            }
            if (view.bias != nullptr) value += view.bias[out];
            y[row * ldy + out] = value;
        });
    });
}

sycl::event linear_fp32_cooperative(sycl::queue& queue, const LinearView& view, const float* x, int64_t ldx,
                                    float* y, int64_t ldy, int64_t rows, const std::vector<sycl::event>& deps) {
    constexpr size_t kSubgroup = 32;
    const size_t groups = static_cast<size_t>(rows) * static_cast<size_t>(view.n);
    return queue.submit([&](sycl::handler& h) {
        if (!deps.empty()) h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(groups * kSubgroup, kSubgroup), [=](sycl::nd_item<1> item)
                       [[sycl::reqd_sub_group_size(32)]] {
            const sycl::sub_group subgroup = item.get_sub_group();
            const size_t group = static_cast<size_t>(item.get_group_linear_id());
            const int64_t row = static_cast<int64_t>(group / static_cast<size_t>(view.n));
            const int64_t out = static_cast<int64_t>(group % static_cast<size_t>(view.n));
            const int lane = static_cast<int>(subgroup.get_local_linear_id());
            const size_t base = static_cast<size_t>(out) * static_cast<size_t>(view.k);
            const float* xr = x + row * ldx;
            float value = 0.0f;
            for (int64_t in = lane; in < view.k; in += static_cast<int64_t>(kSubgroup)) {
                float weight;
                if (view.dtype == DType::F16)
                    weight = f32_from_f16(static_cast<const uint16_t*>(view.data)[base + static_cast<size_t>(in)]);
                else if (view.dtype == DType::BF16)
                    weight = f32_from_bf16(static_cast<const uint16_t*>(view.data)[base + static_cast<size_t>(in)]);
                else
                    weight = static_cast<const float*>(view.data)[base + static_cast<size_t>(in)];
                value += xr[in] * weight;
            }
            value = sycl::reduce_over_group(subgroup, value, sycl::plus<float>());
            if (lane == 0) y[row * ldy + out] = value + (view.bias == nullptr ? 0.0f : view.bias[out]);
        });
    });
}

/// How a half is turned into a float.
enum class HalfMode : int {
    Software,    ///< the value is built from the bits alone
    Native,      ///< the device converts halves, but the tensor may hold exceptional halves
    NativePlain, ///< the device converts halves and the tensor holds only normal halves
};

/// The exact f16 -> f32 conversion of `f32_from_f16`, without a data dependent branch.
///
/// A branch on the exponent of the loaded half costs the whole memory pipeline: the branch waits
/// for its half, so the loads of the other halves of the unrolled loop cannot be in flight.  The
/// exceptional halves (zero, subnormal, infinity, NaN) are therefore selected, not branched on.
///
/// `Native` and `NativePlain` use the device's conversion instruction, which is exact for a normal
/// half; `Software` builds every value from the bits, for a device without half arithmetic.
template <HalfMode Mode>
inline float half_to_float(uint16_t bits) {
    const uint32_t sign = (static_cast<uint32_t>(bits) & 0x8000u) << 16;
    const uint32_t exponent = (static_cast<uint32_t>(bits) >> 10) & 0x1fu;
    const uint32_t mantissa = static_cast<uint32_t>(bits) & 0x3ffu;
    // Infinity and NaN: the exponent field is 255 and the implicit one is dropped.
    const uint32_t infinite = sign | 0x7f800000u | (mantissa << 13);
    // Zero and subnormal: the value is mantissa * 2^-24, which is exact as a float.
    const float scaled = (float) mantissa * 5.9604644775390625e-08f;
    uint32_t scaled_bits;
    std::memcpy(&scaled_bits, &scaled, sizeof scaled_bits);
    const uint32_t subnormal = (scaled_bits & 0x7fffffffu) | sign;
    uint32_t value;
    if constexpr (Mode == HalfMode::NativePlain) {
        // The tensor holds normal halves and signed zeros only, so the device's conversion is
        // exact for every one of them.
        return static_cast<float>(sycl::bit_cast<sycl::half>(bits));
    } else if constexpr (Mode == HalfMode::Native) {
        const float hardware = static_cast<float>(sycl::bit_cast<sycl::half>(bits));
        uint32_t converted;
        std::memcpy(&converted, &hardware, sizeof converted);
        value = exponent == 31u ? infinite : converted;
        value = exponent == 0u ? subnormal : value;
    } else {
        // Normal: the exponent field grows by 112 and the mantissa by 13.
        const uint32_t normal = sign | ((exponent + 112u) << 23) | (mantissa << 13);
        value = exponent == 31u ? infinite : normal;
        value = exponent == 0u ? subnormal : value;
    }
    float result;
    std::memcpy(&result, &value, sizeof result);
    return result;
}

template <DType Type, HalfMode Mode>
inline float raw_weight(const void* data, size_t index) {
    float weight;
    if constexpr (Type == DType::F16) {
        const uint16_t bits = static_cast<const uint16_t*>(data)[index];
        weight = half_to_float<Mode>(bits);
    } else if constexpr (Type == DType::BF16)
        weight = f32_from_bf16(static_cast<const uint16_t*>(data)[index]);
    else
        weight = static_cast<const float*>(data)[index];
    return weight;
}

// Two output rows per wave.  The lanes read the input row once and keep one accumulator per row,
// so the input is fetched once for two rows instead of once for every row.  The input of a wide
// projection is read as often as the projection has outputs, and it is that traffic, not the
// weights, that keeps the one-row-per-wave kernel away from the memory bound.
//
// The accumulation of a row is unchanged: lane l adds the products of the positions l, l+32, l+64
// ... in that order, and the 32 lanes are then reduced over the subgroup.  The result is the same
// float as the one-row-per-wave kernel computes.
template <DType Type, HalfMode Mode>
sycl::event linear_fp32_rows(sycl::queue& queue, const LinearView& view, const float* x, int64_t ldx, float* y,
                              int64_t ldy, int64_t rows, const std::vector<sycl::event>& deps) {
    constexpr size_t kSubgroup = 32;
    constexpr size_t kRows = 2;
    const size_t per_row = (static_cast<size_t>(view.n) + kRows - 1) / kRows;
    const size_t groups = per_row;
    return queue.submit([&](sycl::handler& h) {
        if (!deps.empty()) h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(groups * kSubgroup, kSubgroup), [=](sycl::nd_item<1> item)
                       [[sycl::reqd_sub_group_size(32)]] {
            const sycl::sub_group subgroup = item.get_sub_group();
            const size_t group = static_cast<size_t>(item.get_group_linear_id());
            const size_t first = group * kRows;
            const size_t second = first + 1 < static_cast<size_t>(view.n) ? first + 1 : first;
            const int lane = static_cast<int>(subgroup.get_local_linear_id());
            const size_t base0 = first * static_cast<size_t>(view.k);
            const size_t base1 = second * static_cast<size_t>(view.k);
            // The groups cover the outputs of one input row and the rows are walked here, so that
            // the group index needs no division: a division on a GPU is a long sequence of
            // instructions, and it would be on the path of every group.
            for (int64_t row = 0; row < rows; ++row) {
                const float* xr = x + row * ldx;
                float value0 = 0.0f, value1 = 0.0f;
                #pragma unroll 4
                for (int64_t in = lane; in < view.k; in += static_cast<int64_t>(kSubgroup)) {
                    const float input = xr[in];
                    const size_t offset = static_cast<size_t>(in);
                    value0 += input * raw_weight<Type, Mode>(view.data, base0 + offset);
                    value1 += input * raw_weight<Type, Mode>(view.data, base1 + offset);
                }
                const float reduced0 = sycl::reduce_over_group(subgroup, value0, sycl::plus<float>());
                const float reduced1 = sycl::reduce_over_group(subgroup, value1, sycl::plus<float>());
                if (lane == 0) {
                    y[row * ldy + static_cast<int64_t>(first)] =
                        reduced0 + (view.bias == nullptr ? 0.0f : view.bias[static_cast<int64_t>(first)]);
                    if (second != first)
                        y[row * ldy + static_cast<int64_t>(second)] =
                            reduced1 + (view.bias == nullptr ? 0.0f : view.bias[static_cast<int64_t>(second)]);
                }
            }
        });
    });
}

// For a small number of outputs, parallelize product loads without changing the original
// 32-lane accumulation streams. Products are ephemeral local memory, never expanded weights.
template <DType Type, HalfMode Mode>
sycl::event linear_fp32_products(sycl::queue& queue, const LinearView& view, const float* x, int64_t ldx,
                                 float* y, int64_t ldy, int64_t rows, size_t threads,
                                 const std::vector<sycl::event>& deps) {
    const size_t groups = static_cast<size_t>(rows) * static_cast<size_t>(view.n);
    return queue.submit([&](sycl::handler& h) {
        if (!deps.empty()) h.depends_on(deps);
        sycl::local_accessor<float, 1> products(sycl::range<1>(static_cast<size_t>(view.k)), h);
        h.parallel_for(sycl::nd_range<1>(groups * threads, threads), [=](sycl::nd_item<1> item)
                       [[sycl::reqd_sub_group_size(32)]] {
            const size_t group = item.get_group_linear_id();
            const int64_t row = static_cast<int64_t>(group / static_cast<size_t>(view.n));
            const int64_t out = static_cast<int64_t>(group % static_cast<size_t>(view.n));
            const int64_t t = static_cast<int64_t>(item.get_local_id(0));
            const float* xr = x + row * ldx;
            const size_t base = static_cast<size_t>(out) * static_cast<size_t>(view.k);
            for (int64_t in = t; in < view.k; in += static_cast<int64_t>(threads)) {
                const size_t offset = static_cast<size_t>(in);
                const float weight = raw_weight<Type, Mode>(view.data, base + offset);
                products[offset] = xr[in] * weight;
            }
            item.barrier(sycl::access::fence_space::local_space);
            if (t < 32) {
                float value = 0.0f;
                for (int64_t in = t; in < view.k; in += 32) value += products[static_cast<size_t>(in)];
                value = sycl::reduce_over_group(item.get_sub_group(), value, sycl::plus<float>());
                if (t == 0) y[row * ldy + out] = value + (view.bias == nullptr ? 0.0f : view.bias[out]);
            }
        });
    });
}

template <DType Type, HalfMode Mode>
sycl::event select_raw_kernel(sycl::queue& queue, const LinearView& view, const float* x, int64_t ldx,
                              float* y, int64_t ldy, int64_t rows, const std::vector<sycl::event>& deps) {
    const auto device = queue.get_device();
    const uint64_t max_waves = std::min<uint64_t>(device.get_info<sycl::info::device::max_work_group_size>() / 32,
                                                 device.get_info<sycl::info::device::max_num_sub_groups>());
    const uint64_t capacity = static_cast<uint64_t>(device.get_info<sycl::info::device::max_compute_units>()) * max_waves;
    const uint64_t groups = static_cast<uint64_t>(rows) * static_cast<uint64_t>(view.n);
    // The two-rows-per-wave kernel needs enough output rows to keep the device busy; below that
    // the input range is split over the waves of a work-group instead, and the products travel
    // through local memory.
    const uint64_t wave_groups = (groups + 1) / 2;
    if (wave_groups * 8 >= capacity)
        return linear_fp32_rows<Type, Mode>(queue, view, x, ldx, y, ldy, rows, deps);
    const uint64_t required = capacity / groups + (capacity % groups != 0 ? 1 : 0);
    const uint64_t useful = static_cast<uint64_t>(view.k / 32 + (view.k % 32 != 0 ? 1 : 0));
    const size_t waves = static_cast<size_t>(std::min({max_waves, required, useful}));
    if (waves > 1 && static_cast<uint64_t>(view.k) <= device.get_info<sycl::info::device::local_mem_size>() / sizeof(float))
        return linear_fp32_products<Type, Mode>(queue, view, x, ldx, y, ldy, rows, waves * 32, deps);
    return linear_fp32_rows<Type, Mode>(queue, view, x, ldx, y, ldy, rows, deps);
}

}  // namespace

bool plain_f16_halves(sycl::queue& queue, const void* data, size_t bytes) {
    const size_t count = bytes / 2;
    if (count == 0) return true;
    int* flag = sycl::malloc_device<int>(1, queue);
    if (flag == nullptr) return false;
    const uint16_t* weights = static_cast<const uint16_t*>(data);
    bool plain = false;
    {
        queue.memset(flag, 0, sizeof(int)).wait();
        queue.parallel_for(sycl::range<1>(count), [=](sycl::id<1> id) {
            const uint16_t bits = weights[id[0]];
            const uint32_t magnitude = static_cast<uint32_t>(bits) & 0x7fffu;
            // A plain half is a normal half or a signed zero.  A subnormal half and an infinity
            // or NaN are the ones the device's conversion may not reproduce exactly.
            const bool normal = magnitude == 0u || (magnitude >= 0x0400u && magnitude <= 0x7bffu);
            if (!normal)
                sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>(*flag)
                    .fetch_or(1);
        }).wait();
        int found = 0;
        queue.memcpy(&found, flag, sizeof(int)).wait();
        plain = found == 0;
    }
    sycl::free(flag, queue.get_context());
    return plain;
}

sycl::event linear_fp32(sycl::queue& queue, const LinearView& view, const float* x, int64_t ldx, float* y,
                        int64_t ldy, int64_t rows, const std::vector<sycl::event>& deps) {
    validate(view, x, y, ldx, ldy, rows);
    const uint64_t groups = static_cast<uint64_t>(rows) * static_cast<uint64_t>(view.n);
    if (groups > std::numeric_limits<size_t>::max() / 32 || !cooperative_capable(queue))
        return linear_fp32_scalar(queue, view, x, ldx, y, ldy, rows, deps);
    const char* text = std::getenv("STRATA_EXL3_RAW_SPECIALIZED");
    if (text != nullptr && text[0] == '0' && text[1] == '\0')
        return linear_fp32_cooperative(queue, view, x, ldx, y, ldy, rows, deps);
    switch (view.dtype) {
        case DType::F16:
            if (queue.get_device().has(sycl::aspect::fp16)) {
                if (view.plain_halves)
                    return select_raw_kernel<DType::F16, HalfMode::NativePlain>(queue, view, x, ldx, y, ldy, rows, deps);
                return select_raw_kernel<DType::F16, HalfMode::Native>(queue, view, x, ldx, y, ldy, rows, deps);
            }
            return select_raw_kernel<DType::F16, HalfMode::Software>(queue, view, x, ldx, y, ldy, rows, deps);
        case DType::BF16:
            return select_raw_kernel<DType::BF16, HalfMode::Software>(queue, view, x, ldx, y, ldy, rows, deps);
        default:
            return select_raw_kernel<DType::F32, HalfMode::Software>(queue, view, x, ldx, y, ldy, rows, deps);
    }
}

void validate_embedding(const void* table, DType dtype, size_t row_bytes, const int32_t* tokens, int64_t n_tok,
                        int64_t n_vocab, int64_t n_embd, float* out, int64_t out_stride) {
    if (table == nullptr || tokens == nullptr || out == nullptr)
        throw std::invalid_argument("raw::embedding_gather: table, tokens and output are required");
    if (n_tok <= 0 || n_vocab <= 0 || n_embd <= 0 || out_stride < n_embd)
        throw std::invalid_argument("raw::embedding_gather: invalid dimensions");
    const int bytes = dtype_bytes(dtype);
    if (bytes == 0 || (uint64_t) n_embd > (uint64_t) std::numeric_limits<size_t>::max() / (size_t) bytes ||
        row_bytes < (size_t) n_embd * (size_t) bytes || row_bytes % (size_t) bytes != 0 ||
        reinterpret_cast<uintptr_t>(table) % (size_t) bytes != 0 ||
        reinterpret_cast<uintptr_t>(tokens) % alignof(int32_t) != 0 ||
        reinterpret_cast<uintptr_t>(out) % alignof(float) != 0)
        throw std::invalid_argument("raw::embedding_gather: invalid dtype or row stride");
    const uint64_t max_bytes = (uint64_t) std::numeric_limits<ptrdiff_t>::max();
    if ((uint64_t) n_vocab > max_bytes / row_bytes || (uint64_t) n_tok > max_bytes / sizeof(int32_t) ||
        (uint64_t) n_tok > max_bytes / sizeof(float) / (uint64_t) out_stride)
        throw std::invalid_argument("raw::embedding_gather: span exceeds host address range");
}

void validate_embedding_one(const void* table, DType dtype, size_t row_bytes, int64_t token, int64_t n_vocab,
                            int64_t n_embd, float* out) {
    if (table == nullptr || out == nullptr || token < 0 || token >= n_vocab || n_vocab <= 0 || n_embd <= 0)
        throw std::invalid_argument("raw::embedding_gather_one: invalid token or buffers");
    const int bytes = dtype_bytes(dtype);
    if (bytes == 0 || (uint64_t) n_embd > (uint64_t) std::numeric_limits<size_t>::max() / (size_t) bytes ||
        (uint64_t) n_embd > (uint64_t) std::numeric_limits<ptrdiff_t>::max() / sizeof(float) ||
        row_bytes < (size_t) n_embd * (size_t) bytes || row_bytes % (size_t) bytes != 0 ||
        reinterpret_cast<uintptr_t>(table) % (size_t) bytes != 0 ||
        reinterpret_cast<uintptr_t>(out) % alignof(float) != 0)
        throw std::invalid_argument("raw::embedding_gather_one: invalid dtype or row stride");
    if ((uint64_t) n_vocab > (uint64_t) std::numeric_limits<ptrdiff_t>::max() / row_bytes)
        throw std::invalid_argument("raw::embedding_gather_one: table span exceeds host address range");
}

void embedding_gather(const void* table, DType dtype, size_t row_bytes, const int32_t* tokens, int64_t n_tok,
                      int64_t n_vocab, int64_t n_embd, float* out, int64_t out_stride, void* stream) {
    validate_embedding(table, dtype, row_bytes, tokens, n_tok, n_vocab, n_embd, out, out_stride);
    sycl::queue& queue = queue_for(stream);
    const auto event = queue.parallel_for(sycl::range<1>((size_t) n_tok * (size_t) n_embd), [=](sycl::id<1> id) {
        const size_t flat = (size_t) id[0];
        const int64_t row = (int64_t) (flat / (size_t) n_embd);
        const int64_t col = (int64_t) (flat % (size_t) n_embd);
        const int64_t token = (int64_t) tokens[row];
        float value = 0.0f;
        if (token >= 0 && token < n_vocab) {
            const uint8_t* source = static_cast<const uint8_t*>(table) + (size_t) token * row_bytes;
            if (dtype == DType::F16)
                value = f32_from_f16(static_cast<const uint16_t*>(static_cast<const void*>(source))[col]);
            else if (dtype == DType::BF16)
                value = f32_from_bf16(static_cast<const uint16_t*>(static_cast<const void*>(source))[col]);
            else
                value = static_cast<const float*>(static_cast<const void*>(source))[col];
        }
        out[(size_t) row * (size_t) out_stride + (size_t) col] = value;
    });
    sync_if_needed(stream, event, "raw::embedding_gather");
}

void embedding_gather_one(const void* table, DType dtype, size_t row_bytes, int64_t token, int64_t n_vocab,
                          int64_t n_embd, float* out, void* stream) {
    validate_embedding_one(table, dtype, row_bytes, token, n_vocab, n_embd, out);
    sycl::queue& queue = queue_for(stream);
    const auto event = queue.parallel_for(sycl::range<1>((size_t) n_embd), [=](sycl::id<1> id) {
        const size_t col = (size_t) id[0];
        const uint8_t* source = static_cast<const uint8_t*>(table) + (size_t) token * row_bytes;
        if (dtype == DType::F16)
            out[col] = f32_from_f16(static_cast<const uint16_t*>(static_cast<const void*>(source))[col]);
        else if (dtype == DType::BF16)
            out[col] = f32_from_bf16(static_cast<const uint16_t*>(static_cast<const void*>(source))[col]);
        else
            out[col] = static_cast<const float*>(static_cast<const void*>(source))[col];
    });
    sync_if_needed(stream, event, "raw::embedding_gather_one");
}

}  // namespace strata::kernels::raw
