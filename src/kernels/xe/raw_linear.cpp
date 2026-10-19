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
#include <algorithm>
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

}  // namespace

sycl::event linear_fp32(sycl::queue& queue, const LinearView& view, const float* x, int64_t ldx, float* y,
                        int64_t ldy, int64_t rows, const std::vector<sycl::event>& deps) {
    validate(view, x, y, ldx, ldy, rows);
    const uint64_t groups = static_cast<uint64_t>(rows) * static_cast<uint64_t>(view.n);
    if (groups > std::numeric_limits<size_t>::max() / 32 || !cooperative_capable(queue))
        return linear_fp32_scalar(queue, view, x, ldx, y, ldy, rows, deps);
    return linear_fp32_cooperative(queue, view, x, ldx, y, ldy, rows, deps);
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
