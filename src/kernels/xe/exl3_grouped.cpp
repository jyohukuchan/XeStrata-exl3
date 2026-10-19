// SPDX-FileCopyrightText: 2025 Turboderp
// SPDX-FileCopyrightText: 2026 jyohukuchan
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// EXL3 packed tiles follow exllamav3's MIT-licensed trellis storage rules.
// A bounded, graph-capturable grouped routed expert correctness path.
#include "strata/kernels/exl3_grouped.hpp"

#include "strata/kernels/f16_bits.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_exl3_grouped
#include "strata/kernels/device_arch_tag.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
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
    const int w = word % word_count;
    return (uint32_t) packed[2 * w] | ((uint32_t) packed[2 * w + 1] << 16);
}

inline uint16_t extract_state(const uint16_t* packed, int bits, int state) {
    const int word_count = bits * 8;
    const int b0 = state * bits + bits - 16 + 256 * bits;
    const int b1 = b0 + 16;
    const int i0 = b0 / 32;
    const int i1 = (b1 - 1) / 32;
    const int shift = (i1 + 1) * 32 - b1;
    const uint64_t merged = ((uint64_t) load_word32(packed, i0, word_count) << 32) |
                            (uint64_t) load_word32(packed, i1, word_count);
    return (uint16_t) ((merged >> shift) & 0xFFFFu);
}

inline uint32_t lop3_6a(uint32_t value) {
    return ((~value & 0x3B603B60u) | (value & (0x8FFF8FFFu ^ 0x3B603B60u)));
}

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
    const uint32_t byte_sum = (x & 0xFFu) + ((x >> 8) & 0xFFu) + ((x >> 16) & 0xFFu) + ((x >> 24) & 0xFFu);
    const uint16_t h = (uint16_t) (0x6400u + byte_sum);
    return f16_from_f32(f32_from_f16(h) * f32_from_f16(0x1EEEu) + f32_from_f16(0xC931u));
}

inline float decoded_weight(const MatrixView& view, int64_t row, int64_t col) {
    const int tile_row = (int) (row >> 4);
    const int tile_col = (int) (col >> 4);
    const size_t tile_index = (size_t) tile_row * (size_t) (view.n >> 4) + (size_t) tile_col;
    const uint16_t* tile = view.trellis + tile_index * (size_t) (16 * view.bits);
    const int state = tile_state_for_element((int) ((row & 15) * 16 + (col & 15)));
    return f32_from_f16(decode_state(extract_state(tile, view.bits, state), view.codebook));
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

inline bool span_ok(const ExpertComponentLayout& component, uint64_t total, uint64_t expected, bool required) {
    if (component.bytes == 0) return !required;
    return component.bytes == expected && component.offset % alignof(uint16_t) == 0 &&
           component.offset <= total && component.bytes <= total - component.offset;
}

inline bool projection_ok(const ExpertProjectionLayout& p, int64_t expected_k, int64_t expected_n,
                          bool input_hadamard, bool output_hadamard, uint64_t total) {
    if (p.k != expected_k || p.n != expected_n || p.k <= 0 || p.n <= 0 || p.k % 16 != 0 || p.n % 16 != 0 ||
        p.bits < 1 || p.bits > 8 || p.codebook < 0 || p.codebook > 2)
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

inline MatrixView matrix_from_layout(const ExpertProjectionLayout& p, const uint8_t* base) {
    MatrixView view;
    view.trellis = (const uint16_t*) (base + p.trellis.offset);
    view.suh = p.suh.bytes == 0 ? nullptr : (const uint16_t*) (base + p.suh.offset);
    view.svh = p.svh.bytes == 0 ? nullptr : (const uint16_t*) (base + p.svh.offset);
    view.bias = p.bias.bytes == 0 ? nullptr : (const uint16_t*) (base + p.bias.offset);
    view.k = p.k;
    view.n = p.n;
    view.bits = p.bits;
    view.codebook = p.codebook;
    return view;
}

struct ResolvedEntry {
    MatrixView gate;
    MatrixView up;
    MatrixView down;
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
    result.gate = matrix_from_layout(layout.gate, base);
    result.up = matrix_from_layout(layout.up, base);
    result.down = matrix_from_layout(layout.down, base);
    result.ff = ff;
    result.valid = true;
    return result;
}

inline float dot_value(const MatrixView& view, const float* input_basis, int64_t col) {
    float dot = 0.0f;
    for (int64_t row = 0; row < view.k; ++row) dot += input_basis[row] * decoded_weight(view, row, col);
    return dot;
}

sycl::event submit_gate_up_input(sycl::queue& queue, const GroupedPlan& plan, const float* x, int64_t ldx,
                                 float* scratch, int64_t plane, GroupedOptions options,
                                 const std::vector<sycl::event>& deps) {
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
            gate_dot[col] = dot_value(resolved.gate, gate_input, col);
            up_dot[col] = dot_value(resolved.up, up_input, col);
        });
    });
}

sycl::event submit_gate_up_post(sycl::queue& queue, const GroupedPlan& plan, float* scratch, int64_t plane,
                                GroupedOptions options, const std::vector<sycl::event>& deps) {
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
            const auto post = [&](const MatrixView& view, const GemvOptions& projection_options, const float* dot,
                                  float* output) {
                float value = dot[col];
                if (projection_options.output_hadamard && view.svh != nullptr) {
                    const int64_t block = col & ~int64_t(127);
                    const int local = (int) (col & 127);
                    value = 0.0f;
                    for (int j = 0; j < 128; ++j) value += (float) hadamard_sign(local, j) * dot[block + j];
                    value *= kHadamardScale * f32_from_f16(view.svh[col]);
                }
                if (projection_options.add_bias && view.bias != nullptr) value += f32_from_f16(view.bias[col]);
                output[col] = value;
            };
            post(resolved.gate, options.gate, gate_dot, gate_output);
            post(resolved.up, options.up, up_dot, up_output);
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
    const sycl::event gate_up_input = submit_gate_up_input(queue, plan, x, ldx, scratch, plane, options, deps);
    const sycl::event gate_up_dot = submit_gate_up_dot(queue, plan, scratch, plane, options, {gate_up_input});
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
    const sycl::event down_input = queue.submit([&](sycl::handler& h) {
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
            if (options.down.input_hadamard && resolved.down.suh != nullptr) {
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
    const sycl::event down_dot = queue.submit([&](sycl::handler& h) {
        h.depends_on(down_input);
        h.parallel_for(sycl::range<1>((size_t) plan.max_entries * (size_t) plan.hidden), [=](sycl::id<1> id) {
            const size_t flat = (size_t) id[0];
            const int32_t entry = (int32_t) (flat / (size_t) plan.hidden);
            const int64_t col = (int64_t) (flat % (size_t) plan.hidden);
            const ResolvedEntry resolved = resolve_entry(plan, entry, options);
            if (!resolved.valid) return;
            const float* input = scratch + (size_t) entry * (size_t) plane;
            float* output = scratch + 2 * (size_t) plan.max_entries * (size_t) plane +
                            (size_t) entry * (size_t) plane;
            output[col] = dot_value(resolved.down, input, col);
        });
    });
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
            if (options.down.output_hadamard && resolved.down.svh != nullptr) {
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
                if (options.down.add_bias && resolved.down.bias != nullptr)
                    output[col] += f32_from_f16(resolved.down.bias[col]);
            }
        });
    });
}

}  // namespace strata::kernels::exl3
