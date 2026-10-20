// SPDX-FileCopyrightText: 2025 Turboderp
// SPDX-FileCopyrightText: 2026 jyohukuchan
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// EXL3 packed tiles follow exllamav3's MIT-licensed trellis storage rules.
// A bounded, graph-capturable grouped routed expert correctness path.
#include "strata/kernels/exl3_grouped.hpp"

#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "cuda_intrinsics.hpp"
#include "exl3_cb2.hpp"
#include "exl3_funnel.hpp"
#include "exl3_word_tile.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_exl3_grouped
#include "strata/kernels/device_arch_tag.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::kernels::exl3 {
namespace {

constexpr float kHadamardScale = 0.08838834764831845f;
// gitleaks:allow -- public EXL3 codebook multipliers and bit masks, never credentials.

uint64_t max_float_elements() {
    return (uint64_t) std::numeric_limits<ptrdiff_t>::max() / sizeof(float);
}

void checked_product(int64_t a, int64_t b, uint64_t& result, const char* what) {
    if (a <= 0 || b <= 0 || (uint64_t) a > std::numeric_limits<uint64_t>::max() / (uint64_t) b)
        throw std::invalid_argument(std::string("exl3_grouped_fp32: ") + what + " overflows");
    result = (uint64_t) a * (uint64_t) b;
}

void check_plan(const GroupedPlan& plan, const float* x, int64_t ldx, const float* y, int64_t ldy,
                float* scratch, size_t scratch_elements) {
    if (plan.blob_base == nullptr || plan.format_index == nullptr || plan.format_table == nullptr || plan.start == nullptr ||
        plan.tok == nullptr || plan.dst == nullptr || plan.ngroups == nullptr || plan.nentries == nullptr)
        throw std::invalid_argument("exl3_grouped_fp32: all device plan pointers are required");
    if (plan.max_groups <= 0 || plan.max_groups > std::numeric_limits<int32_t>::max() ||
        plan.max_entries <= 0 || plan.max_entries > std::numeric_limits<int32_t>::max() ||
        plan.max_formats <= 0 || plan.max_formats > std::numeric_limits<int32_t>::max())
        throw std::invalid_argument("exl3_grouped_fp32: capacities must be positive INT32-sized values");
    if (plan.n_tokens <= 0 || plan.output_rows <= 0 || plan.hidden <= 0 || plan.hidden > std::numeric_limits<int32_t>::max() ||
        plan.hidden % 128 != 0 || plan.max_ff <= 0 || plan.max_ff > std::numeric_limits<int32_t>::max())
        throw std::invalid_argument("exl3_grouped_fp32: invalid static geometry");
    if (x == nullptr || y == nullptr || scratch == nullptr)
        throw std::invalid_argument("exl3_grouped_fp32: x, y and scratch are required");
    if (ldx < plan.hidden || ldy < plan.hidden)
        throw std::invalid_argument("exl3_grouped_fp32: leading dimensions are smaller than hidden size");
    const uint64_t span_limit = max_float_elements();
    if ((uint64_t) plan.n_tokens > span_limit / (uint64_t) ldx ||
        (uint64_t) plan.output_rows > span_limit / (uint64_t) ldy)
        throw std::invalid_argument("exl3_grouped_fp32: input/output spans exceed host address range");
    const int64_t scratch_width = std::max(plan.max_ff, plan.hidden);
    uint64_t scratch_plane = 0;
    checked_product(plan.max_entries, scratch_width, scratch_plane, "scratch plane size");
    if (scratch_plane > span_limit / 6 || scratch_plane * 6 > std::numeric_limits<size_t>::max())
        throw std::invalid_argument("exl3_grouped_fp32: scratch size exceeds host address range");
    if (scratch_elements < (size_t) (scratch_plane * 6))
        throw std::invalid_argument("exl3_grouped_fp32: caller scratch is too small");
}

inline uint32_t load_word32(const uint16_t* packed, int word, int word_count) {
    // extract_state visits [word_count - 1, 2 * word_count - 1], so one ring wrap suffices.
    const int w = word >= word_count ? word - word_count : word;
    return (uint32_t) packed[2 * w] | ((uint32_t) packed[2 * w + 1] << 16);
}

inline uint16_t extract_state(const uint16_t* packed, int bits, int state) {
    const int word_count = bits * 8;
    const int b0 = state * bits + bits - 16 + 256 * bits;
    const int b1 = b0 + 16;
    const int i0 = b0 / 32;
    const int i1 = (b1 - 1) / 32;
    const int shift = (i1 + 1) * 32 - b1;
    const uint32_t hi = load_word32(packed, i0, word_count);
    const uint32_t lo = load_word32(packed, i1, word_count);
    return static_cast<uint16_t>(strata::kernels::xe::funnel_right_u32(hi, lo, static_cast<unsigned>(shift)));
}

inline uint32_t lop3_6a(uint32_t value) {
    return ((~value & 0x3B603B60u) | (value & (0x8FFF8FFFu ^ 0x3B603B60u)));
}

// The codebook 2 decode has two forms, the portable one and the half fused multiply add of the reference
// decoder; they return the same F16 bits (exl3_cb2.hpp).  HalfFma selects the form at compile time, so the
// kernels carry no runtime branch in the decode.
template <bool HalfFma>
inline uint16_t decode_state(uint16_t state, int codebook) {
    uint32_t x = state;
    if (codebook == 0) {
        x = x * 89226354u + 64248484u;
        x = lop3_6a(x);
        return f16_from_f32(f32_from_f16((uint16_t) x) + f32_from_f16((uint16_t) (x >> 16)));
    }
    if (codebook == 1) {
        x = x * 0xCBAC1FEDu;
        x = lop3_6a(x);
        return f16_from_f32(f32_from_f16((uint16_t) x) + f32_from_f16((uint16_t) (x >> 16)));
    }
    x = x * 0x83DCD12Du;
    const int byte_sum = cb2_byte_sum(x);
    return cb2_value_bits<HalfFma>(byte_sum);
}

template <bool HalfFma>
inline float decode_state_f32(uint16_t state, int codebook) {
    if (codebook != 2) return f32_from_f16(decode_state<HalfFma>(state, codebook));
    const uint32_t x = state * 0x83DCD12Du;
    const int byte_sum = cb2_byte_sum(x);
    return cb2_value_f32<HalfFma>(byte_sum);
}

template <bool HalfFma>
inline float decoded_weight(const MatrixView& view, int64_t row, int64_t col) {
    const int tile_row = (int) (row >> 4);
    const int tile_col = (int) (col >> 4);
    const size_t tile_index = (size_t) tile_row * (size_t) (view.n >> 4) + (size_t) tile_col;
    const uint16_t* tile = view.trellis + tile_index * (size_t) (16 * view.bits);
    const int state = tile_state_for_element((int) ((row & 15) * 16 + (col & 15)));
    return decode_state_f32<HalfFma>(extract_state(tile, view.bits, state), view.codebook);
}

// The caller has already checked that this packed matrix uses codebook 2.  Keep bits runtime
// and preserve the exact trellis extraction; the literal codebook lets the compiler fold decode_state_f32's
// non-CB2 path without changing the accumulation order or tile/padding guards.
template <bool HalfFma>
inline float decoded_weight_cb2(const MatrixView& view, int64_t row, int64_t col) {
    const int tile_row = (int) (row >> 4);
    const int tile_col = (int) (col >> 4);
    const size_t tile_index = (size_t) tile_row * (size_t) (view.n >> 4) + (size_t) tile_col;
    const uint16_t* tile = view.trellis + tile_index * (size_t) (16 * view.bits);
    const int state = tile_state_for_element((int) ((row & 15) * 16 + (col & 15)));
    return decode_state_f32<HalfFma>(extract_state(tile, view.bits, state), 2);
}

// Two K lanes per output retain two independent accumulators (K mod 4), preserving the
// previous four-lane addition order while sharing all sixteen output columns of a packed tile.
struct GroupedPartial { float even = 0.0f; float odd = 0.0f; };

template <int Bits, int RR = 0, bool HalfFma = false>
inline void accumulate_grouped_tile(const word_cache::DistributedTile<Bits>& words,
                                    const sycl::sub_group& subgroup, int lane, const float* input,
                                    int64_t row_base, GroupedPartial& sum) {
    const int row = 2 * RR + (lane >> 4);
    const float weight = decode_state_f32<HalfFma>(words.template state<RR>(subgroup, lane), 2);
    if constexpr (RR % 2 == 0) sum.even += input[row_base + row] * weight;
    else sum.odd += input[row_base + row] * weight;
    if constexpr (RR < 7)
        accumulate_grouped_tile<Bits, RR + 1, HalfFma>(words, subgroup, lane, input, row_base, sum);
}

template <int Bits, bool HalfFma>
inline GroupedPartial grouped_word_dot(const MatrixView& view, const float* input, int64_t col,
                                       int64_t row_begin, int64_t row_end,
                                       const sycl::sub_group& subgroup, int lane) {
    GroupedPartial sum;
    for (int64_t row = row_begin; row < row_end; row += 16) {
        const uint16_t* tile = view.trellis +
            (static_cast<size_t>(row / 16) * static_cast<size_t>(view.n / 16) +
             static_cast<size_t>(col / 16)) * static_cast<size_t>(16 * Bits);
        word_cache::DistributedTile<Bits> words{0, 0};
        if (lane < Bits * 8) words.bank0 = load_word32(tile, lane, Bits * 8);
        if constexpr (Bits > 4)
            if (lane + 32 < Bits * 8) words.bank1 = load_word32(tile, lane + 32, Bits * 8);
        accumulate_grouped_tile<Bits, 0, HalfFma>(words, subgroup, lane, input, row, sum);
    }
    return sum;
}

template <bool HalfFma>
inline GroupedPartial grouped_word_dot_dispatch(const MatrixView& view, const float* input, int64_t col,
                                                int64_t row_begin, int64_t row_end,
                                                const sycl::sub_group& subgroup, int lane) {
    switch (view.bits) {
#define STRATA_GROUPED_WORD_CASE(B) \
        case B: return grouped_word_dot<B, HalfFma>(view, input, col, row_begin, row_end, subgroup, lane)
        STRATA_GROUPED_WORD_CASE(1);
        STRATA_GROUPED_WORD_CASE(2);
        STRATA_GROUPED_WORD_CASE(3);
        STRATA_GROUPED_WORD_CASE(4);
        STRATA_GROUPED_WORD_CASE(5);
        STRATA_GROUPED_WORD_CASE(6);
        STRATA_GROUPED_WORD_CASE(7);
        STRATA_GROUPED_WORD_CASE(8);
#undef STRATA_GROUPED_WORD_CASE
        default: return {}; // A validated packed descriptor always has bits in [1, 8].
    }
}

// Fallback for other codebooks and K partitions not aligned to a whole compressed tile.
template <bool CB2, bool HalfFma>
inline GroupedPartial grouped_scalar_partial_impl(const MatrixView& view, const float* input, int64_t col,
                                             int64_t iter_begin, int64_t iter_end, int k_lane, bool paired) {
    GroupedPartial sum;
    for (int64_t iter = iter_begin; iter < iter_end; ++iter) {
        const int64_t row = iter * 4 + k_lane;
        sum.even += input[row] * (CB2 ? decoded_weight_cb2<HalfFma>(view, row, col)
                                      : decoded_weight<HalfFma>(view, row, col));
        if (paired)
            sum.odd += input[row + 2] * (CB2 ? decoded_weight_cb2<HalfFma>(view, row + 2, col)
                                            : decoded_weight<HalfFma>(view, row + 2, col));
    }
    return sum;
}

template <bool HalfFma>
inline GroupedPartial grouped_scalar_partial(const MatrixView& view, const float* input, int64_t col,
                                             int64_t iter_begin, int64_t iter_end, int k_lane, bool paired) {
    if (view.codebook == 2)
        return grouped_scalar_partial_impl<true, HalfFma>(view, input, col, iter_begin, iter_end, k_lane, paired);
    return grouped_scalar_partial_impl<false, HalfFma>(view, input, col, iter_begin, iter_end, k_lane, paired);
}

inline float reduce_grouped_partial(const GroupedPartial& sum, const sycl::sub_group& subgroup, bool paired) {
    if (paired) {
        const float first = sum.even + sycl::permute_group_by_xor(subgroup, sum.even, 16);
        const float second = sum.odd + sycl::permute_group_by_xor(subgroup, sum.odd, 16);
        return first + second;
    }
    float value = sum.even + sycl::permute_group_by_xor(subgroup, sum.even, 1);
    return value + sycl::permute_group_by_xor(subgroup, value, 2);
}

bool grouped_word_tile_enabled() {
    const char* text = std::getenv("STRATA_EXL3_GROUPED_WORD_TILE");
    return text == nullptr || !(text[0] == '0' && text[1] == '\0');
}

inline bool valid_raw_dtype(raw::DType dtype) {
    return dtype == raw::DType::F16 || dtype == raw::DType::BF16 || dtype == raw::DType::F32;
}

inline float raw_value(const RawProjectionView& view, size_t index) {
    if (view.dtype == raw::DType::F16) return f32_from_f16(static_cast<const uint16_t*>(view.data)[index]);
    if (view.dtype == raw::DType::BF16) return f32_from_bf16(static_cast<const uint16_t*>(view.data)[index]);
    return static_cast<const float*>(view.data)[index];
}

inline float raw_bias_value(const RawProjectionView& view, size_t index) {
    if (view.bias_dtype == raw::DType::F16) return f32_from_f16(static_cast<const uint16_t*>(view.bias)[index]);
    if (view.bias_dtype == raw::DType::BF16) return f32_from_bf16(static_cast<const uint16_t*>(view.bias)[index]);
    return static_cast<const float*>(view.bias)[index];
}

inline int hadamard_sign(int row, int col) {
    unsigned bits = (unsigned) (row & col);
    int parity = 0;
    while (bits != 0) {
        parity ^= 1;
        bits &= bits - 1;
    }
    return parity == 0 ? 1 : -1;
}

bool grouped_cooperative_capability(const sycl::queue& queue);

inline void fwht128_registers(const sycl::sub_group& sg, int lane, float& a0, float& a1, float& a2, float& a3) {
    for (int step = 1; step < 32; step <<= 1) {
        const float p0 = sycl::permute_group_by_xor(sg, a0, step);
        const float p1 = sycl::permute_group_by_xor(sg, a1, step);
        const float p2 = sycl::permute_group_by_xor(sg, a2, step);
        const float p3 = sycl::permute_group_by_xor(sg, a3, step);
        a0 = (lane & step) ? p0 - a0 : a0 + p0;
        a1 = (lane & step) ? p1 - a1 : a1 + p1;
        a2 = (lane & step) ? p2 - a2 : a2 + p2;
        a3 = (lane & step) ? p3 - a3 : a3 + p3;
    }
    const float b0 = a0 + a1, b1 = a0 - a1, b2 = a2 + a3, b3 = a2 - a3;
    a0 = b0 + b2; a1 = b1 + b3; a2 = b0 - b2; a3 = b1 - b3;
}

inline void fwht128_input(const sycl::sub_group& sg, int lane, const float* input, const uint16_t* suh,
                          int64_t block, float& a0, float& a1, float& a2, float& a3) {
    a0 = input[block + lane] * f32_from_f16(suh[block + lane]);
    a1 = input[block + lane + 32] * f32_from_f16(suh[block + lane + 32]);
    a2 = input[block + lane + 64] * f32_from_f16(suh[block + lane + 64]);
    a3 = input[block + lane + 96] * f32_from_f16(suh[block + lane + 96]);
    fwht128_registers(sg, lane, a0, a1, a2, a3);
    a0 *= kHadamardScale; a1 *= kHadamardScale; a2 *= kHadamardScale; a3 *= kHadamardScale;
}

inline void fwht128_output(const sycl::sub_group& sg, int lane, const float* dot, const uint16_t* svh,
                           const uint16_t* bias, int64_t block, bool add_bias, float& a0, float& a1, float& a2,
                           float& a3) {
    a0 = dot[block + lane]; a1 = dot[block + lane + 32];
    a2 = dot[block + lane + 64]; a3 = dot[block + lane + 96];
    fwht128_registers(sg, lane, a0, a1, a2, a3);
    a0 *= kHadamardScale * f32_from_f16(svh[block + lane]);
    a1 *= kHadamardScale * f32_from_f16(svh[block + lane + 32]);
    a2 *= kHadamardScale * f32_from_f16(svh[block + lane + 64]);
    a3 *= kHadamardScale * f32_from_f16(svh[block + lane + 96]);
    if (add_bias && bias != nullptr) {
        a0 += f32_from_f16(bias[block + lane]); a1 += f32_from_f16(bias[block + lane + 32]);
        a2 += f32_from_f16(bias[block + lane + 64]); a3 += f32_from_f16(bias[block + lane + 96]);
    }
}

inline bool span_ok(const ExpertComponentLayout& component, uint64_t total, uint64_t expected, bool required) {
    if (component.bytes == 0) return !required;
    return component.bytes == expected && component.offset % alignof(uint16_t) == 0 &&
           component.offset <= total && component.bytes <= total - component.offset;
}

inline bool projection_ok(const ExpertProjectionLayout& p, int64_t expected_k, int64_t expected_n,
                          bool input_hadamard, bool output_hadamard, uint64_t total) {
    if (p.k != expected_k || p.n != expected_n || p.k <= 0 || p.n <= 0) return false;
    if (p.storage == ProjectionStorage::raw) {
        if (!valid_raw_dtype(p.raw_dtype) || !valid_raw_dtype(p.raw_bias_dtype) || p.bits != 0 || p.codebook != 0 ||
            p.raw_weights.bytes == 0 || p.trellis.bytes != 0 || p.suh.bytes != 0 || p.svh.bytes != 0 || p.bias.bytes != 0)
            return false;
        const uint64_t k = (uint64_t) p.k, n = (uint64_t) p.n;
        if (k > std::numeric_limits<uint64_t>::max() / n) return false;
        const uint64_t elements = k * n;
        const uint64_t weight_size = (uint64_t) raw::dtype_bytes(p.raw_dtype);
        if (elements > std::numeric_limits<uint64_t>::max() / weight_size) return false;
        const uint64_t weights = elements * weight_size;
        const uint64_t bias_size = (uint64_t) raw::dtype_bytes(p.raw_bias_dtype);
        if (n > std::numeric_limits<uint64_t>::max() / bias_size) return false;
        const uint64_t bias = p.raw_bias.bytes == 0 ? 0 : n * bias_size;
        return span_ok(p.raw_weights, total, weights, true) && span_ok(p.raw_bias, total, bias, false);
    }
    if (p.storage != ProjectionStorage::packed_trellis || p.k % 16 != 0 || p.n % 16 != 0 || p.bits < 1 || p.bits > 8 ||
        p.codebook < 0 || p.codebook > 2 || p.raw_weights.bytes != 0 || p.raw_bias.bytes != 0)
        return false;
    const uint64_t k = (uint64_t) p.k, n = (uint64_t) p.n, bits = (uint64_t) p.bits;
    if (k > std::numeric_limits<uint64_t>::max() / n) return false;
    const uint64_t elements = k * n;
    if (elements > std::numeric_limits<uint64_t>::max() / bits) return false;
    const uint64_t trellis = elements * bits / 8;
    if (!span_ok(p.trellis, total, trellis, true) || !span_ok(p.suh, total, k * 2, false) ||
        !span_ok(p.svh, total, n * 2, false) || !span_ok(p.bias, total, n * 2, false))
        return false;
    if ((input_hadamard && p.suh.bytes != 0 && p.k % 128 != 0) ||
        (output_hadamard && p.svh.bytes != 0 && p.n % 128 != 0))
        return false;
    return true;
}

inline ProjectionView matrix_from_layout(const ExpertProjectionLayout& p, const uint8_t* base) {
    if (p.storage == ProjectionStorage::raw) {
        RawProjectionView raw_view;
        raw_view.data = base + p.raw_weights.offset;
        raw_view.bias = p.raw_bias.bytes == 0 ? nullptr : base + p.raw_bias.offset;
        raw_view.k = p.k;
        raw_view.n = p.n;
        raw_view.dtype = p.raw_dtype;
        raw_view.bias_dtype = p.raw_bias_dtype;
        return ProjectionView::from_raw(raw_view);
    }
    MatrixView view;
    view.trellis = (const uint16_t*) (base + p.trellis.offset);
    view.suh = p.suh.bytes == 0 ? nullptr : (const uint16_t*) (base + p.suh.offset);
    view.svh = p.svh.bytes == 0 ? nullptr : (const uint16_t*) (base + p.svh.offset);
    view.bias = p.bias.bytes == 0 ? nullptr : (const uint16_t*) (base + p.bias.offset);
    view.k = p.k;
    view.n = p.n;
    view.bits = p.bits;
    view.codebook = p.codebook;
    // The device blob is the canonical one: the CPU block order never reaches
    // the device, so the tile order stays the format's own.
    view.tile_order = TileOrder::canonical;
    return ProjectionView(view);
}

struct ResolvedEntry {
    ProjectionView gate{MatrixView{}};
    ProjectionView up{MatrixView{}};
    ProjectionView down{MatrixView{}};
    int32_t token = -1;
    int32_t destination = -1;
    int64_t ff = 0;
    bool valid = false;
    bool has_destination = false;
};

inline ResolvedEntry resolve_entry(const GroupedPlan& plan, int32_t entry, const GroupedOptions& options) {
    ResolvedEntry result;
    const int32_t groups = *plan.ngroups;
    const int32_t entries = *plan.nentries;
    if (groups < 0 || groups > plan.max_groups || entries < 0 || entries > plan.max_entries || entry < 0 || entry >= entries)
        return result;
    int32_t group = -1;
    for (int32_t g = 0; g < groups; ++g) {
        const int32_t first = plan.start[g], last = plan.start[g + 1];
        if (first < 0 || last < first || last > entries) return result;
        if (entry >= first && entry < last) {
            group = g;
            break;
        }
    }
    if (group < 0) return result;
    result.token = plan.tok[entry];
    result.destination = plan.dst[entry];
    result.has_destination = result.destination >= 0 && result.destination < plan.output_rows;
    if (result.token < 0 || result.token >= plan.n_tokens) {
        result.has_destination = false;  // invalid input IDs are skipped rather than used to emit a sentinel
        return result;
    }
    const int32_t format = plan.format_index[group];
    if (format < 0 || format >= plan.max_formats || plan.blob_base[group] == 0 ||
        plan.blob_base[group] % alignof(uint16_t) != 0 || !result.has_destination) return result;
    const ExpertBlobLayout layout = plan.format_table[format];
    const int64_t ff = layout.gate.n;
    if (layout.total_bytes == 0 || layout.total_bytes > (uint64_t) std::numeric_limits<ptrdiff_t>::max() ||
        plan.blob_base[group] > std::numeric_limits<uint64_t>::max() - layout.total_bytes ||
        layout.gate.k != plan.hidden || layout.up.k != plan.hidden ||
        layout.gate.n != layout.up.n || layout.down.k != ff || layout.down.n != plan.hidden || ff <= 0 ||
        ff > plan.max_ff || !projection_ok(layout.gate, plan.hidden, ff, options.gate.input_hadamard,
                                            options.gate.output_hadamard, layout.total_bytes) ||
        !projection_ok(layout.up, plan.hidden, ff, options.up.input_hadamard, options.up.output_hadamard,
                       layout.total_bytes) ||
        !projection_ok(layout.down, ff, plan.hidden, options.down.input_hadamard,
                       options.down.output_hadamard, layout.total_bytes))
        return result;
    const uint8_t* base = (const uint8_t*) plan.blob_base[group];
    const auto aligned_raw_projection = [base](const ExpertProjectionLayout& projection) {
        if (projection.storage != ProjectionStorage::raw) return true;
        const uint64_t weight_alignment = (uint64_t) raw::dtype_bytes(projection.raw_dtype);
        const uint64_t bias_alignment = (uint64_t) raw::dtype_bytes(projection.raw_bias_dtype);
        if (weight_alignment == 0 || (projection.raw_weights.offset + (uint64_t) (uintptr_t) base) % weight_alignment != 0)
            return false;
        return projection.raw_bias.bytes == 0 ||
               (bias_alignment != 0 && (projection.raw_bias.offset + (uint64_t) (uintptr_t) base) % bias_alignment == 0);
    };
    if (!aligned_raw_projection(layout.gate) || !aligned_raw_projection(layout.up) ||
        !aligned_raw_projection(layout.down))
        return result;
    result.gate = matrix_from_layout(layout.gate, base);
    result.up = matrix_from_layout(layout.up, base);
    result.down = matrix_from_layout(layout.down, base);
    result.ff = ff;
    result.valid = true;
    return result;
}

template <bool HalfFma>
inline float dot_value(const ProjectionView& view, const float* input_basis, int64_t col) {
    float dot = 0.0f;
    if (view.is_packed()) {
        const MatrixView packed = view.packed_matrix();
        for (int64_t row = 0; row < view.k; ++row) dot += input_basis[row] * decoded_weight<HalfFma>(packed, row, col);
    } else {
        const size_t base = (size_t) col * (size_t) view.k;
        for (int64_t row = 0; row < view.k; ++row) dot += input_basis[row] * raw_value(view.raw, base + (size_t) row);
    }
    return dot;
}

sycl::event submit_gate_up_input_cooperative(sycl::queue& queue, const GroupedPlan& plan, const float* x,
                                             int64_t ldx, float* scratch, int64_t plane, GroupedOptions options,
                                             const std::vector<sycl::event>& deps) {
    const size_t blocks = static_cast<size_t>(plan.hidden / 128);
    return queue.submit([&](sycl::handler& h) {
        if (!deps.empty()) h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(plan.max_entries) * blocks * 32), 32),
                       [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
            const auto sg = item.get_sub_group(); const int lane = (int) sg.get_local_id()[0];
            const size_t g = (size_t) item.get_group_linear_id();
            const int32_t entry = (int32_t) (g / blocks); const int64_t block = (int64_t) (g % blocks) * 128;
            const ResolvedEntry r = resolve_entry(plan, entry, options); const bool active = r.valid;
            const float* in = active ? x + (int64_t) r.token * ldx : nullptr;
            float* gate = active ? scratch + (size_t) entry * (size_t) plane : nullptr;
            float* up = active ? scratch + (size_t) plan.max_entries * (size_t) plane + (size_t) entry * (size_t) plane : nullptr;
            float g0=0,g1=0,g2=0,g3=0,u0=0,u1=0,u2=0,u3=0;
            if (active && options.gate.input_hadamard && r.gate.suh) fwht128_input(sg,lane,in,r.gate.suh,block,g0,g1,g2,g3);
            else if (active) { g0=in[block+lane];g1=in[block+lane+32];g2=in[block+lane+64];g3=in[block+lane+96]; }
            if (active && options.up.input_hadamard && r.up.suh) fwht128_input(sg,lane,in,r.up.suh,block,u0,u1,u2,u3);
            else if (active) { u0=in[block+lane];u1=in[block+lane+32];u2=in[block+lane+64];u3=in[block+lane+96]; }
            if (active) { gate[block+lane]=g0;gate[block+lane+32]=g1;gate[block+lane+64]=g2;gate[block+lane+96]=g3;
                          up[block+lane]=u0;up[block+lane+32]=u1;up[block+lane+64]=u2;up[block+lane+96]=u3; }
        });
    });
}

sycl::event submit_gate_up_post_cooperative(sycl::queue& queue, const GroupedPlan& plan, float* scratch, int64_t plane,
                                            GroupedOptions options, const std::vector<sycl::event>& deps) {
    const size_t blocks = (static_cast<size_t>(plan.max_ff) + 127u) / 128u;
    return queue.submit([&](sycl::handler& h) {
        if (!deps.empty()) h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(static_cast<size_t>(plan.max_entries) * blocks * 32, 32),
                       [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
            const auto sg=item.get_sub_group(); const int lane=(int)sg.get_local_id()[0]; const size_t g=(size_t)item.get_group_linear_id();
            const int32_t entry = static_cast<int32_t>(g / blocks);
            const int64_t block = static_cast<int64_t>(g % blocks) * 128;
            const ResolvedEntry r = resolve_entry(plan, entry, options);
            if (!r.valid || block >= r.ff) return;
            float* gd = scratch + 2 * (size_t) plan.max_entries * (size_t) plane +
                        (size_t) entry * (size_t) plane;
            float* ud = scratch + 3 * (size_t) plan.max_entries * (size_t) plane +
                        (size_t) entry * (size_t) plane;
            float* go = scratch + 4 * (size_t) plan.max_entries * (size_t) plane +
                        (size_t) entry * (size_t) plane;
            float* uo = scratch + 5 * (size_t) plan.max_entries * (size_t) plane +
                        (size_t) entry * (size_t) plane;
            const bool gh = options.gate.output_hadamard && r.gate.svh;
            const bool uh = options.up.output_hadamard && r.up.svh;
            float g0 = 0, g1 = 0, g2 = 0, g3 = 0;
            float u0 = 0, u1 = 0, u2 = 0, u3 = 0;
            const int64_t c0 = block + lane, c1 = c0 + 32, c2 = c0 + 64, c3 = c0 + 96;
            if (gh) {
                fwht128_output(sg, lane, gd, r.gate.svh, r.gate.bias, block, options.gate.add_bias,
                               g0, g1, g2, g3);
            } else {
                if (c0 < r.ff) g0 = gd[c0]; if (c1 < r.ff) g1 = gd[c1];
                if (c2 < r.ff) g2 = gd[c2]; if (c3 < r.ff) g3 = gd[c3];
            }
            if (uh) {
                fwht128_output(sg, lane, ud, r.up.svh, r.up.bias, block, options.up.add_bias,
                               u0, u1, u2, u3);
            } else {
                if (c0 < r.ff) u0 = ud[c0]; if (c1 < r.ff) u1 = ud[c1];
                if (c2 < r.ff) u2 = ud[c2]; if (c3 < r.ff) u3 = ud[c3];
            }
            if (options.gate.add_bias && !gh) {
                if (r.gate.is_packed() && r.gate.bias) {
                    if (c0 < r.ff) g0 += f32_from_f16(r.gate.bias[c0]);
                    if (c1 < r.ff) g1 += f32_from_f16(r.gate.bias[c1]);
                    if (c2 < r.ff) g2 += f32_from_f16(r.gate.bias[c2]);
                    if (c3 < r.ff) g3 += f32_from_f16(r.gate.bias[c3]);
                } else if (r.gate.is_raw() && r.gate.raw.bias) {
                    if (c0 < r.ff) g0 += raw_bias_value(r.gate.raw, (size_t) c0);
                    if (c1 < r.ff) g1 += raw_bias_value(r.gate.raw, (size_t) c1);
                    if (c2 < r.ff) g2 += raw_bias_value(r.gate.raw, (size_t) c2);
                    if (c3 < r.ff) g3 += raw_bias_value(r.gate.raw, (size_t) c3);
                }
            }
            if (options.up.add_bias && !uh) {
                if (r.up.is_packed() && r.up.bias) {
                    if (c0 < r.ff) u0 += f32_from_f16(r.up.bias[c0]);
                    if (c1 < r.ff) u1 += f32_from_f16(r.up.bias[c1]);
                    if (c2 < r.ff) u2 += f32_from_f16(r.up.bias[c2]);
                    if (c3 < r.ff) u3 += f32_from_f16(r.up.bias[c3]);
                } else if (r.up.is_raw() && r.up.raw.bias) {
                    if (c0 < r.ff) u0 += raw_bias_value(r.up.raw, (size_t) c0);
                    if (c1 < r.ff) u1 += raw_bias_value(r.up.raw, (size_t) c1);
                    if (c2 < r.ff) u2 += raw_bias_value(r.up.raw, (size_t) c2);
                    if (c3 < r.ff) u3 += raw_bias_value(r.up.raw, (size_t) c3);
                }
            }
            if (c0 < r.ff) { go[c0] = g0; uo[c0] = u0; }
            if (c1 < r.ff) { go[c1] = g1; uo[c1] = u1; }
            if (c2 < r.ff) { go[c2] = g2; uo[c2] = u2; }
            if (c3 < r.ff) { go[c3] = g3; uo[c3] = u3; }
        });
    });
}

sycl::event submit_gate_up_input(sycl::queue& queue, const GroupedPlan& plan, const float* x, int64_t ldx,
                                 float* scratch, int64_t plane, GroupedOptions options,
                                 const std::vector<sycl::event>& deps) {
    if (grouped_cooperative_capability(queue))
        return submit_gate_up_input_cooperative(queue, plan, x, ldx, scratch, plane, options, deps);
    return queue.submit([&](sycl::handler& h) {
        if (!deps.empty()) h.depends_on(deps);
        h.parallel_for(sycl::range<1>((size_t) plan.max_entries * (size_t) plan.hidden), [=](sycl::id<1> id) {
            const size_t flat = (size_t) id[0];
            const int32_t entry = (int32_t) (flat / (size_t) plan.hidden);
            const int64_t row = (int64_t) (flat % (size_t) plan.hidden);
            const ResolvedEntry resolved = resolve_entry(plan, entry, options);
            if (!resolved.valid) return;
            const float* input = x + (int64_t) resolved.token * ldx;
            float* gate_input = scratch + (size_t) entry * (size_t) plane;
            float* up_input = scratch + (size_t) plan.max_entries * (size_t) plane +
                              (size_t) entry * (size_t) plane;
            const int64_t gate_block = row & ~int64_t(127);
            const int gate_local = (int) (row & 127);
            const int64_t up_block = row & ~int64_t(127);
            const int up_local = (int) (row & 127);
            if (options.gate.input_hadamard && resolved.gate.suh != nullptr) {
                float sum = 0.0f;
                for (int j = 0; j < 128; ++j)
                    sum += (float) hadamard_sign(gate_local, j) * input[gate_block + j] *
                           f32_from_f16(resolved.gate.suh[gate_block + j]);
                gate_input[row] = sum * kHadamardScale;
            } else {
                gate_input[row] = input[row];
            }
            if (options.up.input_hadamard && resolved.up.suh != nullptr) {
                float sum = 0.0f;
                for (int j = 0; j < 128; ++j)
                    sum += (float) hadamard_sign(up_local, j) * input[up_block + j] *
                           f32_from_f16(resolved.up.suh[up_block + j]);
                up_input[row] = sum * kHadamardScale;
            } else {
                up_input[row] = input[row];
            }
        });
    });
}

bool grouped_cooperative_capability(const sycl::queue& queue) {
    try {
        const auto subgroup_sizes = queue.get_device().get_info<sycl::info::device::sub_group_sizes>();
        if (std::find(subgroup_sizes.begin(), subgroup_sizes.end(), size_t(32)) == subgroup_sizes.end()) return false;
        return queue.get_device().get_info<sycl::info::device::max_work_group_size>() >= 32;
    } catch (...) {
        return false;
    }
}

int gate_up_split_k(const sycl::queue& queue, const GroupedPlan& plan) {
    if (plan.hidden <= 0 || (plan.hidden & 3) != 0) return 1;
    try {
        const auto device = queue.get_device();
        const size_t max_waves = device.get_info<sycl::info::device::max_work_group_size>() / 32;
        const size_t subgroup_limit = device.get_info<sycl::info::device::max_num_sub_groups>();
        // Each wave reserves two padded 16-column planes for the eight gate/up partials.
        const size_t local_waves = device.get_info<sycl::info::device::local_mem_size>() / (2 * 16 * sizeof(float));
        const size_t cap = std::max<size_t>(1, std::min({max_waves, subgroup_limit, local_waves}));
        int split = static_cast<int>(std::min<size_t>(static_cast<size_t>(plan.hidden / 4), cap));
        // An explicit upper bound supports per-device tuning without embedding a one-device threshold.
        const char* text = std::getenv("STRATA_EXL3_GU_SPLIT_K");
        if (text != nullptr) {
            char* end = nullptr;
            const long requested = std::strtol(text, &end, 10);
            if (end != text && *end == '\0' && requested > 0)
                split = static_cast<int>(std::min<long>(split, requested));
        }
        return std::max(1, split);
    } catch (...) {
        return 1;
    }
}

template <bool HalfFma>
sycl::event submit_gate_up_dot_cooperative(sycl::queue& queue, const GroupedPlan& plan, float* scratch,
                                           int64_t plane, GroupedOptions options,
                                           const std::vector<sycl::event>& deps) {
    const int split_k = gate_up_split_k(queue, plan);
    const bool word_tile = grouped_word_tile_enabled();
    const size_t columns = word_tile ? 16u : 8u;
    const size_t tiles = (static_cast<size_t>(plan.max_ff) + columns - 1) / columns;
    const size_t groups = static_cast<size_t>(plan.max_entries) * tiles;
    const size_t local_size = static_cast<size_t>(split_k) * 32u;
    const int64_t k_iters = plan.hidden / 4;
    return queue.submit([&](sycl::handler& h) {
        if (!deps.empty()) h.depends_on(deps);
        sycl::local_accessor<float, 1> reduce(sycl::range<1>(static_cast<size_t>(split_k) * 32u), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(groups * local_size), sycl::range<1>(local_size)),
                       [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
            const sycl::sub_group subgroup = item.get_sub_group();
            const int lane = static_cast<int>(subgroup.get_local_id()[0]);
            const int wave = static_cast<int>(item.get_local_id(0)) / 32;
            const int column_lane = word_tile ? lane & 15 : lane >> 2;
            const int k_lane = word_tile ? lane >> 4 : lane & 3;
            const size_t group = static_cast<size_t>(item.get_group_linear_id());
            const int32_t entry = static_cast<int32_t>(group / tiles);
            const int64_t tile = static_cast<int64_t>(group % tiles);
            const int64_t col = tile * static_cast<int64_t>(columns) + column_lane;
            const ResolvedEntry resolved = resolve_entry(plan, entry, options);
            const bool active = resolved.valid && col < resolved.ff;
            const float* gate_input = active ? scratch + static_cast<size_t>(entry) * static_cast<size_t>(plane) : nullptr;
            const float* up_input = active ? scratch + static_cast<size_t>(plan.max_entries) * static_cast<size_t>(plane) +
                                                   static_cast<size_t>(entry) * static_cast<size_t>(plane) : nullptr;
            const int64_t iter_begin = (k_iters * wave) / split_k;
            const int64_t iter_end = (k_iters * (wave + 1)) / split_k;
            GroupedPartial gate_partial;
            GroupedPartial up_partial;
            if (active) {
                if (resolved.gate.is_packed()) {
                    const MatrixView gate = resolved.gate.packed_matrix();
                    if (gate.codebook == 2 && word_tile && iter_begin % 4 == 0 && iter_end % 4 == 0) {
                        gate_partial = grouped_word_dot_dispatch<HalfFma>(gate, gate_input, col, iter_begin * 4,
                                                                         iter_end * 4, subgroup, lane);
                    } else {
                        gate_partial = grouped_scalar_partial<HalfFma>(gate, gate_input, col, iter_begin,
                                                                     iter_end, k_lane, word_tile);
                    }
                } else if (wave == 0 && k_lane == 0) {
                    // Preserve the existing raw/mixed projection summation order; only packed EXL3 dots are K-split.
                    gate_partial.even = dot_value<HalfFma>(resolved.gate, gate_input, col);
                }
                if (resolved.up.is_packed()) {
                    const MatrixView up = resolved.up.packed_matrix();
                    if (up.codebook == 2 && word_tile && iter_begin % 4 == 0 && iter_end % 4 == 0) {
                        up_partial = grouped_word_dot_dispatch<HalfFma>(up, up_input, col, iter_begin * 4,
                                                                       iter_end * 4, subgroup, lane);
                    } else {
                        up_partial = grouped_scalar_partial<HalfFma>(up, up_input, col, iter_begin,
                                                                   iter_end, k_lane, word_tile);
                    }
                } else if (wave == 0 && k_lane == 0) {
                    // Preserve the existing raw/mixed projection summation order; only packed EXL3 dots are K-split.
                    up_partial.even = dot_value<HalfFma>(resolved.up, up_input, col);
                }
            }
            const float gate_reduced = reduce_grouped_partial(gate_partial, subgroup, word_tile);
            const float up_reduced = reduce_grouped_partial(up_partial, subgroup, word_tile);
            if (active && k_lane == 0) {
                reduce[static_cast<size_t>(wave) * 16u + static_cast<size_t>(column_lane)] = gate_reduced;
                reduce[static_cast<size_t>(split_k) * 16u + static_cast<size_t>(wave) * 16u +
                       static_cast<size_t>(column_lane)] = up_reduced;
            }
            item.barrier();
            if (active && wave == 0 && k_lane == 0) {
                float gate_sum = 0.0f;
                float up_sum = 0.0f;
                for (int w = 0; w < split_k; ++w) {
                    gate_sum += reduce[static_cast<size_t>(w) * 16u + static_cast<size_t>(column_lane)];
                    up_sum += reduce[static_cast<size_t>(split_k) * 16u + static_cast<size_t>(w) * 16u +
                                      static_cast<size_t>(column_lane)];
                }
                float* gate_dot = scratch + 2 * static_cast<size_t>(plan.max_entries) * static_cast<size_t>(plane) +
                                  static_cast<size_t>(entry) * static_cast<size_t>(plane);
                float* up_dot = scratch + 3 * static_cast<size_t>(plan.max_entries) * static_cast<size_t>(plane) +
                                static_cast<size_t>(entry) * static_cast<size_t>(plane);
                gate_dot[col] = gate_sum;
                up_dot[col] = up_sum;
            }
        });
    });
}

template <bool HalfFma>
sycl::event submit_gate_up_dot(sycl::queue& queue, const GroupedPlan& plan, float* scratch, int64_t plane,
                               GroupedOptions options, const std::vector<sycl::event>& deps) {
    return queue.submit([&](sycl::handler& h) {
        if (!deps.empty()) h.depends_on(deps);
        h.parallel_for(sycl::range<1>((size_t) plan.max_entries * (size_t) plan.max_ff), [=](sycl::id<1> id) {
            const size_t flat = (size_t) id[0];
            const int32_t entry = (int32_t) (flat / (size_t) plan.max_ff);
            const int64_t col = (int64_t) (flat % (size_t) plan.max_ff);
            const ResolvedEntry resolved = resolve_entry(plan, entry, options);
            if (!resolved.valid || col >= resolved.ff) return;
            const float* gate_input = scratch + (size_t) entry * (size_t) plane;
            const float* up_input = scratch + (size_t) plan.max_entries * (size_t) plane +
                                    (size_t) entry * (size_t) plane;
            float* gate_dot = scratch + 2 * (size_t) plan.max_entries * (size_t) plane +
                              (size_t) entry * (size_t) plane;
            float* up_dot = scratch + 3 * (size_t) plan.max_entries * (size_t) plane +
                            (size_t) entry * (size_t) plane;
            gate_dot[col] = dot_value<HalfFma>(resolved.gate, gate_input, col);
            up_dot[col] = dot_value<HalfFma>(resolved.up, up_input, col);
        });
    });
}

sycl::event submit_gate_up_post(sycl::queue& queue, const GroupedPlan& plan, float* scratch, int64_t plane,
                                GroupedOptions options, const std::vector<sycl::event>& deps) {
    if (grouped_cooperative_capability(queue))
        return submit_gate_up_post_cooperative(queue, plan, scratch, plane, options, deps);
    return queue.submit([&](sycl::handler& h) {
        if (!deps.empty()) h.depends_on(deps);
        h.parallel_for(sycl::range<1>((size_t) plan.max_entries * (size_t) plan.max_ff), [=](sycl::id<1> id) {
            const size_t flat = (size_t) id[0];
            const int32_t entry = (int32_t) (flat / (size_t) plan.max_ff);
            const int64_t col = (int64_t) (flat % (size_t) plan.max_ff);
            const ResolvedEntry resolved = resolve_entry(plan, entry, options);
            if (!resolved.valid || col >= resolved.ff) return;
            float* gate_dot = scratch + 2 * (size_t) plan.max_entries * (size_t) plane +
                              (size_t) entry * (size_t) plane;
            float* up_dot = scratch + 3 * (size_t) plan.max_entries * (size_t) plane +
                            (size_t) entry * (size_t) plane;
            float* gate_output = scratch + 4 * (size_t) plan.max_entries * (size_t) plane +
                                 (size_t) entry * (size_t) plane;
            float* up_output = scratch + 5 * (size_t) plan.max_entries * (size_t) plane +
                               (size_t) entry * (size_t) plane;
            const auto post = [&](const ProjectionView& view, const GemvOptions& projection_options, const float* dot,
                                  float* output) {
                float value = dot[col];
                if (view.is_packed() && projection_options.output_hadamard && view.svh != nullptr) {
                    const int64_t block = col & ~int64_t(127);
                    const int local = (int) (col & 127);
                    value = 0.0f;
                    for (int j = 0; j < 128; ++j) value += (float) hadamard_sign(local, j) * dot[block + j];
                    value *= kHadamardScale * f32_from_f16(view.svh[col]);
                }
                if (projection_options.add_bias) {
                    if (view.is_packed() && view.bias != nullptr) value += f32_from_f16(view.bias[col]);
                    else if (view.is_raw() && view.raw.bias != nullptr) value += raw_bias_value(view.raw, (size_t) col);
                }
                output[col] = value;
            };
            post(resolved.gate, options.gate, gate_dot, gate_output);
            post(resolved.up, options.up, up_dot, up_output);
        });
    });
}

template <bool HalfFma>
sycl::event submit_down_dot_cooperative(sycl::queue& queue, const GroupedPlan& plan, float* scratch,
                                         int64_t plane, GroupedOptions options,
                                         const std::vector<sycl::event>& deps) {
    const bool word_tile = grouped_word_tile_enabled();
    const size_t columns = word_tile ? 16u : 8u;
    const size_t tiles = (static_cast<size_t>(plan.hidden) + columns - 1) / columns;
    const size_t groups = static_cast<size_t>(plan.max_entries) * tiles;
    return queue.submit([&](sycl::handler& h) {
        if (!deps.empty()) h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(groups * 32u), sycl::range<1>(32)),
                       [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
            const sycl::sub_group subgroup = item.get_sub_group();
            const int lane = static_cast<int>(subgroup.get_local_id()[0]);
            const int column_lane = word_tile ? lane & 15 : lane >> 2;
            const int k_lane = word_tile ? lane >> 4 : lane & 3;
            const size_t group = static_cast<size_t>(item.get_group_linear_id());
            const int32_t entry = static_cast<int32_t>(group / tiles);
            const int64_t tile = static_cast<int64_t>(group % tiles);
            const int64_t col = tile * static_cast<int64_t>(columns) + column_lane;
            const ResolvedEntry resolved = resolve_entry(plan, entry, options);
            const bool active = resolved.valid && col < plan.hidden;
            const float* input = active ? scratch + static_cast<size_t>(entry) * static_cast<size_t>(plane) : nullptr;
            GroupedPartial partial;
            if (active) {
                if (resolved.down.is_packed()) {
                    const MatrixView down = resolved.down.packed_matrix();
                    if (down.codebook == 2 && word_tile) {
                        partial = grouped_word_dot_dispatch<HalfFma>(down, input, col, 0, down.k, subgroup, lane);
                    } else {
                        partial = grouped_scalar_partial<HalfFma>(down, input, col, 0, down.k / 4, k_lane, word_tile);
                    }
                } else if (k_lane == 0) {
                    partial.even = dot_value<HalfFma>(resolved.down, input, col);
                }
            }
            const float reduced = reduce_grouped_partial(partial, subgroup, word_tile);
            if (active && k_lane == 0) {
                float* output = scratch + 2 * static_cast<size_t>(plan.max_entries) * static_cast<size_t>(plane) +
                                static_cast<size_t>(entry) * static_cast<size_t>(plane);
                output[col] = reduced;
            }
        });
    });
}

sycl::event submit_down_input_cooperative(sycl::queue& queue, const GroupedPlan& plan, float* scratch,
                                          int64_t plane, GroupedOptions options, const std::vector<sycl::event>& deps) {
    const size_t blocks = (static_cast<size_t>(plan.max_ff) + 127u) / 128u;
    return queue.submit([&](sycl::handler& h) {
        if (!deps.empty()) h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(static_cast<size_t>(plan.max_entries) * blocks * 32, 32),
                       [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
            const auto sg = item.get_sub_group();
            const int lane = (int) sg.get_local_id()[0];
            const size_t group = (size_t) item.get_group_linear_id();
            const int32_t entry = (int32_t) (group / blocks);
            const int64_t block = (int64_t) (group % blocks) * 128;
            const ResolvedEntry resolved = resolve_entry(plan, entry, options);
            if (!resolved.valid || block >= resolved.ff) return;
            const float* input = scratch + 4 * (size_t) plan.max_entries * (size_t) plane +
                                 (size_t) entry * (size_t) plane;
            float* output = scratch + (size_t) entry * (size_t) plane;
            float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            if (options.down.input_hadamard && resolved.down.suh) {
                fwht128_input(sg, lane, input, resolved.down.suh, block, a0, a1, a2, a3);
            } else {
                const int64_t c0 = block + lane, c1 = c0 + 32, c2 = c0 + 64, c3 = c0 + 96;
                if (c0 < resolved.ff) a0 = input[c0]; if (c1 < resolved.ff) a1 = input[c1];
                if (c2 < resolved.ff) a2 = input[c2]; if (c3 < resolved.ff) a3 = input[c3];
            }
            const int64_t c0 = block + lane, c1 = c0 + 32, c2 = c0 + 64, c3 = c0 + 96;
            if (c0 < resolved.ff) output[c0] = a0; if (c1 < resolved.ff) output[c1] = a1;
            if (c2 < resolved.ff) output[c2] = a2; if (c3 < resolved.ff) output[c3] = a3;
        });
    });
}

sycl::event submit_down_post_cooperative(sycl::queue& queue, const GroupedPlan& plan, float* scratch, int64_t plane,
                                         int64_t ldy, float* y, GroupedOptions options,
                                         const std::vector<sycl::event>& deps) {
    const size_t blocks = static_cast<size_t>(plan.hidden / 128);
    return queue.submit([&](sycl::handler& h) {
        if (!deps.empty()) h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(static_cast<size_t>(plan.max_entries) * blocks * 32, 32),
                       [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
            const auto sg = item.get_sub_group();
            const int lane = (int) sg.get_local_id()[0];
            const size_t group = (size_t) item.get_group_linear_id();
            const int32_t entry = (int32_t) (group / blocks);
            const int64_t block = (int64_t) (group % blocks) * 128;
            const ResolvedEntry resolved = resolve_entry(plan, entry, options);
            float* output = resolved.has_destination ? y + (int64_t) resolved.destination * ldy : nullptr;
            if (!resolved.valid) {
                if (output) {
                    output[block + lane] = sycl::nan((uint32_t) 0);
                    output[block + lane + 32] = sycl::nan((uint32_t) 0);
                    output[block + lane + 64] = sycl::nan((uint32_t) 0);
                    output[block + lane + 96] = sycl::nan((uint32_t) 0);
                }
                return;
            }
            const float* dot = scratch + 2 * (size_t) plan.max_entries * (size_t) plane +
                               (size_t) entry * (size_t) plane;
            float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            if (options.down.output_hadamard && resolved.down.svh) {
                fwht128_output(sg, lane, dot, resolved.down.svh, resolved.down.bias, block,
                               options.down.add_bias, a0, a1, a2, a3);
            } else {
                const int64_t c0 = block + lane, c1 = c0 + 32, c2 = c0 + 64, c3 = c0 + 96;
                if (c0 < plan.hidden) a0 = dot[c0]; if (c1 < plan.hidden) a1 = dot[c1];
                if (c2 < plan.hidden) a2 = dot[c2]; if (c3 < plan.hidden) a3 = dot[c3];
                if (options.down.add_bias) {
                    if (resolved.down.is_packed() && resolved.down.bias) {
                        if (c0 < plan.hidden) a0 += f32_from_f16(resolved.down.bias[c0]);
                        if (c1 < plan.hidden) a1 += f32_from_f16(resolved.down.bias[c1]);
                        if (c2 < plan.hidden) a2 += f32_from_f16(resolved.down.bias[c2]);
                        if (c3 < plan.hidden) a3 += f32_from_f16(resolved.down.bias[c3]);
                    } else if (resolved.down.is_raw() && resolved.down.raw.bias) {
                        if (c0 < plan.hidden) a0 += raw_bias_value(resolved.down.raw, (size_t) c0);
                        if (c1 < plan.hidden) a1 += raw_bias_value(resolved.down.raw, (size_t) c1);
                        if (c2 < plan.hidden) a2 += raw_bias_value(resolved.down.raw, (size_t) c2);
                        if (c3 < plan.hidden) a3 += raw_bias_value(resolved.down.raw, (size_t) c3);
                    }
                }
            }
            const int64_t c0 = block + lane, c1 = c0 + 32, c2 = c0 + 64, c3 = c0 + 96;
            if (c0 < plan.hidden) output[c0] = a0; if (c1 < plan.hidden) output[c1] = a1;
            if (c2 < plan.hidden) output[c2] = a2; if (c3 < plan.hidden) output[c3] = a3;
        });
    });
}

// The plain down dot, for the devices without the cooperative path.  It has the same two decode forms.
template <bool HalfFma>
sycl::event submit_down_dot(sycl::queue& queue, const GroupedPlan& plan, float* scratch, int64_t plane,
                            GroupedOptions options, const std::vector<sycl::event>& deps) {
    return queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>((size_t) plan.max_entries * (size_t) plan.hidden), [=](sycl::id<1> id) {
            const size_t flat = (size_t) id[0];
            const int32_t entry = (int32_t) (flat / (size_t) plan.hidden);
            const int64_t col = (int64_t) (flat % (size_t) plan.hidden);
            const ResolvedEntry resolved = resolve_entry(plan, entry, options);
            if (!resolved.valid) return;
            const float* input = scratch + (size_t) entry * (size_t) plane;
            float* output = scratch + 2 * (size_t) plan.max_entries * (size_t) plane +
                            (size_t) entry * (size_t) plane;
            output[col] = dot_value<HalfFma>(resolved.down, input, col);
        });
    });
}

}  // namespace

size_t exl3_grouped_scratch_elements(const GroupedPlan& plan) {
    if (plan.max_entries <= 0 || plan.max_ff <= 0 || plan.hidden <= 0)
        throw std::invalid_argument("exl3_grouped_fp32: capacities must be positive");
    const int64_t scratch_width = std::max(plan.max_ff, plan.hidden);
    uint64_t plane = 0;
    checked_product(plan.max_entries, scratch_width, plane, "scratch plane size");
    if (plane > std::numeric_limits<size_t>::max() / 6 || plane > max_float_elements() / 6)
        throw std::invalid_argument("exl3_grouped_fp32: scratch exceeds the host address range");
    return (size_t) plane * 6;
}

sycl::event exl3_grouped_fp32(sycl::queue& queue, const GroupedPlan& plan, const float* x, int64_t ldx, float* y,
                              int64_t ldy, float* scratch, size_t scratch_elements, GroupedOptions options,
                              const std::vector<sycl::event>& deps) {
    check_plan(plan, x, ldx, y, ldy, scratch, scratch_elements);
    const int64_t plane = std::max(plan.max_ff, plan.hidden);
    // The codebook 2 decode runs in its half form on the devices that report fp16 arithmetic, and in its
    // portable form otherwise.  The form is a template parameter, so the kernels carry no runtime branch.
    const bool half_fma = cb2_half_fma_enabled(queue);
    const sycl::event gate_up_input = submit_gate_up_input(queue, plan, x, ldx, scratch, plane, options, deps);
    sycl::event gate_up_dot;
    if (grouped_cooperative_capability(queue)) {
        if (half_fma)
            gate_up_dot = submit_gate_up_dot_cooperative<true>(queue, plan, scratch, plane, options, {gate_up_input});
        else
            gate_up_dot = submit_gate_up_dot_cooperative<false>(queue, plan, scratch, plane, options, {gate_up_input});
    } else {
        if (half_fma)
            gate_up_dot = submit_gate_up_dot<true>(queue, plan, scratch, plane, options, {gate_up_input});
        else
            gate_up_dot = submit_gate_up_dot<false>(queue, plan, scratch, plane, options, {gate_up_input});
    }
    const sycl::event gate_up_post = submit_gate_up_post(queue, plan, scratch, plane, options, {gate_up_dot});
    const sycl::event activation = queue.submit([&](sycl::handler& h) {
        h.depends_on(gate_up_post);
        h.parallel_for(sycl::range<1>((size_t) plan.max_entries * (size_t) plan.max_ff), [=](sycl::id<1> id) {
            const size_t flat = (size_t) id[0];
            const int32_t entry = (int32_t) (flat / (size_t) plan.max_ff);
            const int64_t col = (int64_t) (flat % (size_t) plan.max_ff);
            const ResolvedEntry resolved = resolve_entry(plan, entry, options);
            if (!resolved.valid || col >= resolved.ff) return;
            float* gate_values = scratch + 4 * (size_t) plan.max_entries * (size_t) plane +
                                 (size_t) entry * (size_t) plane;
            const float* up_values = scratch + 5 * (size_t) plan.max_entries * (size_t) plane +
                                     (size_t) entry * (size_t) plane;
            const float g = gate_values[col];
            gate_values[col] = (g / (1.0f + sycl::exp(-g))) * up_values[col];
        });
    });
    sycl::event down_input;
    if (grouped_cooperative_capability(queue)) {
        down_input = submit_down_input_cooperative(queue, plan, scratch, plane, options, {activation});
    } else {
        down_input = queue.submit([&](sycl::handler& h) {
        h.depends_on(activation);
        h.parallel_for(sycl::range<1>((size_t) plan.max_entries * (size_t) plan.max_ff), [=](sycl::id<1> id) {
            const size_t flat = (size_t) id[0];
            const int32_t entry = (int32_t) (flat / (size_t) plan.max_ff);
            const int64_t row = (int64_t) (flat % (size_t) plan.max_ff);
            const ResolvedEntry resolved = resolve_entry(plan, entry, options);
            if (!resolved.valid || row >= resolved.ff) return;
            const float* activation_values = scratch + 4 * (size_t) plan.max_entries * (size_t) plane +
                                             (size_t) entry * (size_t) plane;
            float* down_input_values = scratch + (size_t) entry * (size_t) plane;
            if (resolved.down.is_packed() && options.down.input_hadamard && resolved.down.suh != nullptr) {
                const int64_t block = row & ~int64_t(127);
                const int local = (int) (row & 127);
                float sum = 0.0f;
                for (int j = 0; j < 128; ++j)
                    sum += (float) hadamard_sign(local, j) * activation_values[block + j] *
                           f32_from_f16(resolved.down.suh[block + j]);
                down_input_values[row] = sum * kHadamardScale;
            } else {
                down_input_values[row] = activation_values[row];
            }
        });
        });
    }
    sycl::event down_dot;
    if (grouped_cooperative_capability(queue)) {
        if (half_fma)
            down_dot = submit_down_dot_cooperative<true>(queue, plan, scratch, plane, options, {down_input});
        else
            down_dot = submit_down_dot_cooperative<false>(queue, plan, scratch, plane, options, {down_input});
    } else {
        if (half_fma)
            down_dot = submit_down_dot<true>(queue, plan, scratch, plane, options, {down_input});
        else
            down_dot = submit_down_dot<false>(queue, plan, scratch, plane, options, {down_input});
    }
    if (grouped_cooperative_capability(queue))
        return submit_down_post_cooperative(queue, plan, scratch, plane, ldy, y, options, {down_dot});
    return queue.submit([&](sycl::handler& h) {
        h.depends_on(down_dot);
        h.parallel_for(sycl::range<1>((size_t) plan.max_entries * (size_t) plan.hidden), [=](sycl::id<1> id) {
            const size_t flat = (size_t) id[0];
            const int32_t entry = (int32_t) (flat / (size_t) plan.hidden);
            const int64_t col = (int64_t) (flat % (size_t) plan.hidden);
            const ResolvedEntry resolved = resolve_entry(plan, entry, options);
            if (!resolved.valid) {
                if (resolved.has_destination)
                    y[(int64_t) resolved.destination * ldy + col] = sycl::nan((uint32_t) 0);
                return;
            }
            const float* dot = scratch + 2 * (size_t) plan.max_entries * (size_t) plane +
                               (size_t) entry * (size_t) plane;
            float* output = y + (int64_t) resolved.destination * ldy;
            if (resolved.down.is_packed() && options.down.output_hadamard && resolved.down.svh != nullptr) {
                const int64_t block = col & ~int64_t(127);
                const int local = (int) (col & 127);
                float transformed = 0.0f;
                for (int j = 0; j < 128; ++j) transformed += (float) hadamard_sign(local, j) * dot[block + j];
                transformed *= kHadamardScale * f32_from_f16(resolved.down.svh[col]);
                if (options.down.add_bias && resolved.down.bias != nullptr)
                    transformed += f32_from_f16(resolved.down.bias[col]);
                output[col] = transformed;
            } else {
                output[col] = dot[col];
                if (options.down.add_bias) {
                    if (resolved.down.is_packed() && resolved.down.bias != nullptr)
                        output[col] += f32_from_f16(resolved.down.bias[col]);
                    else if (resolved.down.is_raw() && resolved.down.raw.bias != nullptr)
                        output[col] += raw_bias_value(resolved.down.raw, (size_t) col);
                }
            }
        });
    });
}

}  // namespace strata::kernels::exl3
