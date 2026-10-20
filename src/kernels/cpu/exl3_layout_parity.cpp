// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// The CPU tile orders against the format's own order.  All of them hold the
// same compressed bits, so every result must be bit identical; the test also
// checks that a mixed-up order is actually detected.
#include "strata/artifact/exl3_expert.hpp"
#include "strata/kernels/cpu/exl3.hpp"
#include "strata/kernels/cpu/exl3_expert.hpp"
#include "strata/kernels/cpu/exl3_mul1_intdot.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace cpu_exl3 = strata::kernels::cpu::exl3;
namespace format = strata::kernels::exl3;

namespace {

int failures = 0;

void check(bool ok, const std::string& label) {
    std::printf("  %s: %s\n", label.c_str(), ok ? "OK" : "FAIL");
    if (!ok) ++failures;
}

/// The CPU's own orders, and the format's own order beside them.
const format::TileOrder kCpuOrders[] = {format::TileOrder::cpu_block, format::TileOrder::cpu_nmajor};

const char* order_name(format::TileOrder order) {
    switch (order) {
        case format::TileOrder::cpu_block: return "cpu_block";
        case format::TileOrder::cpu_nmajor: return "cpu_nmajor";
        default: return "canonical";
    }
}

struct Matrix {
    int64_t k = 0;
    int64_t n = 0;
    int bits = 0;
    int codebook = 0;
    std::vector<uint16_t> trellis;
    std::vector<uint16_t> suh;
    std::vector<uint16_t> svh;
    std::vector<uint16_t> bias;
};

uint32_t next_random(uint32_t& seed) {
    seed = seed * 1664525u + 1013904223u;
    return seed >> 8;
}

Matrix make_matrix(int64_t k, int64_t n, int bits, int codebook, uint32_t seed) {
    Matrix matrix;
    matrix.k = k;
    matrix.n = n;
    matrix.bits = bits;
    matrix.codebook = codebook;
    const size_t tiles = static_cast<size_t>(k / 16) * static_cast<size_t>(n / 16);
    matrix.trellis.resize(tiles * static_cast<size_t>(16 * bits));
    for (uint16_t& value : matrix.trellis) value = static_cast<uint16_t>(next_random(seed));
    matrix.suh.resize(static_cast<size_t>(k));
    matrix.svh.resize(static_cast<size_t>(n));
    matrix.bias.resize(static_cast<size_t>(n));
    for (uint16_t& value : matrix.suh) value = 0x3c00;   // 1.0
    for (uint16_t& value : matrix.svh) value = 0x3c00;   // 1.0
    for (uint16_t& value : matrix.bias) value = 0x0000;  // 0.0
    return matrix;
}

/// The same matrix with its compressed tiles in the given CPU order.
Matrix reordered(const Matrix& source, format::TileOrder order) {
    Matrix result = source;
    std::vector<uint16_t> canonical = source.trellis;
    strata::artifact::exl3::copy_packed_tiles(canonical.data(), result.trellis.data(), source.k, source.n, source.bits,
                                              format::TileOrder::canonical, order);
    return result;
}

format::MatrixView view(const Matrix& matrix, format::TileOrder order) {
    format::MatrixView result;
    result.trellis = matrix.trellis.data();
    result.suh = matrix.suh.data();
    result.svh = matrix.svh.data();
    result.bias = matrix.bias.data();
    result.k = matrix.k;
    result.n = matrix.n;
    result.bits = matrix.bits;
    result.codebook = matrix.codebook;
    result.tile_order = order;
    return result;
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) return false;
    return true;
}

bool all_finite(const std::vector<float>& values) {
    for (float value : values) if (!std::isfinite(value)) return false;
    return true;
}

std::vector<float> make_input(int64_t count, uint32_t seed) {
    std::vector<float> x(static_cast<size_t>(count));
    for (size_t i = 0; i < x.size(); ++i) {
        seed = seed * 1664525u + 1013904223u;
        x[i] = 0.03125f * static_cast<float>(static_cast<int>((seed >> 16) % 29) - 14);
    }
    return x;
}

// The packed projection's own results: the W-hat matrix, the whole-matrix
// product and every 128-column block.  The block is the unit of the H128
// transform, so the row-parallel path never asks for less.
void check_matrix(const Matrix& canonical, const Matrix& cpu, format::TileOrder order, const std::vector<float>& x,
                  const std::string& label, size_t& projections, size_t& values) {
    ++projections;
    const format::MatrixView canonical_view = view(canonical, format::TileOrder::canonical);
    const format::MatrixView cpu_view = view(cpu, order);

    std::vector<uint16_t> canonical_w(static_cast<size_t>(canonical.k * canonical.n));
    std::vector<uint16_t> cpu_w(static_cast<size_t>(cpu.k * cpu.n));
    const cpu_exl3::Status canonical_status = cpu_exl3::reconstruct(canonical_view, canonical_w.data(), canonical_w.size());
    const cpu_exl3::Status cpu_status = cpu_exl3::reconstruct(cpu_view, cpu_w.data(), cpu_w.size());
    check(canonical_status == cpu_exl3::Status::ok && cpu_status == cpu_exl3::Status::ok && canonical_w == cpu_w,
          label + " reconstruct is bit identical");

    std::vector<float> canonical_input(static_cast<size_t>(canonical.k));
    std::array<uint16_t, 256> tile{};
    cpu_exl3::LinearScratch canonical_scratch{canonical_input.data(), canonical_input.size(), tile.data(), tile.size()};
    std::vector<float> canonical_y(static_cast<size_t>(canonical.n));
    std::vector<float> cpu_y(static_cast<size_t>(cpu.n));
    const cpu_exl3::Status canonical_linear = cpu_exl3::linear(canonical_view, x.data(), canonical_y.data(), canonical_scratch);
    const cpu_exl3::Status cpu_linear = cpu_exl3::linear(cpu_view, x.data(), cpu_y.data(), canonical_scratch);
    check(canonical_linear == cpu_exl3::Status::ok && cpu_linear == cpu_exl3::Status::ok && same_bits(canonical_y, cpu_y) &&
              all_finite(cpu_y),
          label + " linear is bit identical");
    values += canonical_y.size();

    // The row-parallel path reads one 128-column block at a time.  The output
    // buffer holds the whole output, as the pool's workspace does.
    bool blocks_match = true;
    for (int64_t col = 0; col + 128 <= canonical.n; col += 128) {
        std::vector<float> canonical_output(static_cast<size_t>(canonical.n), 0.0f);
        std::vector<float> cpu_output(static_cast<size_t>(cpu.n), 0.0f);
        std::vector<float> canonical_basis(static_cast<size_t>(canonical.k));
        std::vector<float> cpu_basis(static_cast<size_t>(cpu.k));
        std::vector<float> dot_buffer(static_cast<size_t>(canonical.k) + 128, 0.0f);
        cpu_exl3::LinearScratch block_scratch{dot_buffer.data(), dot_buffer.size(), tile.data(), tile.size()};
        const cpu_exl3::Status canonical_prepare =
            cpu_exl3::prepare_input_basis(canonical_view, x.data(), canonical_basis.data());
        const cpu_exl3::Status cpu_prepare = cpu_exl3::prepare_input_basis(cpu_view, x.data(), cpu_basis.data());
        const cpu_exl3::Status canonical_block_status =
            cpu_exl3::linear_output_block(canonical_view, canonical_basis.data(), col, canonical_output.data(),
                                          block_scratch);
        const cpu_exl3::Status cpu_block_status =
            cpu_exl3::linear_output_block(cpu_view, cpu_basis.data(), col, cpu_output.data(), block_scratch);
        if (canonical_prepare != cpu_exl3::Status::ok || cpu_prepare != cpu_exl3::Status::ok ||
            canonical_block_status != cpu_exl3::Status::ok || cpu_block_status != cpu_exl3::Status::ok)
            blocks_match = false;
        for (int64_t i = 0; i < 128; ++i)
            if (std::memcmp(&canonical_output[static_cast<size_t>(col + i)], &cpu_output[static_cast<size_t>(col + i)],
                            sizeof(float)) != 0)
                blocks_match = false;
    }
    check(blocks_match, label + " output blocks are bit identical");
    values += canonical.n;

    // The integer dot, when it is enabled, must read the same tiles.  It is
    // the MUL1 codebook's path, so only that codebook reaches it.
    if (cpu_exl3::mul1_intdot_enabled() && canonical.codebook == 2) {
        std::vector<int32_t> canonical_duplicated(static_cast<size_t>(canonical.k));
        std::vector<int32_t> cpu_duplicated(static_cast<size_t>(cpu.k));
        cpu_exl3::Mul1Input canonical_prepared;
        cpu_exl3::Mul1Input cpu_prepared;
        const bool canonical_ready = cpu_exl3::prepare_mul1_input(
            canonical_view, x.data(), canonical_duplicated.data(), canonical_duplicated.size(), canonical_prepared);
        const bool cpu_ready =
            cpu_exl3::prepare_mul1_input(cpu_view, x.data(), cpu_duplicated.data(), cpu_duplicated.size(), cpu_prepared);
        bool intdot_match = canonical_ready && cpu_ready;
        if (intdot_match) {
            std::vector<float> canonical_dot(static_cast<size_t>(canonical.n));
            std::vector<float> cpu_dot(static_cast<size_t>(cpu.n));
            const size_t tiles = static_cast<size_t>(canonical.n / 16);
            intdot_match = cpu_exl3::linear_mul1_dot_range(canonical_view, canonical_prepared, 0, tiles,
                                                            canonical_dot.data()) &&
                           cpu_exl3::linear_mul1_dot_range(cpu_view, cpu_prepared, 0, tiles, cpu_dot.data()) &&
                           same_bits(canonical_dot, cpu_dot);
        }
        check(intdot_match, label + " integer dot is bit identical");
    }
}

}  // namespace

int main() {
    // A geometry whose output width covers several 128-column blocks, and one
    // that covers exactly one.
    const struct {
        int64_t hidden;
        int64_t ff;
    } shapes[] = {{128, 128}, {256, 256}, {128, 384}};
    uint32_t seed = 20261020u;
    size_t compared = 0;
    size_t projections = 0;
    size_t values = 0;
    for (const auto& shape : shapes) {
        for (const int bits : {2, 4, 5}) {
            for (const int codebook : {0, 1, 2}) {
                const Matrix canonical_gate = make_matrix(shape.hidden, shape.ff, bits, codebook, seed);
                const Matrix canonical_up = make_matrix(shape.hidden, shape.ff, bits, codebook, seed + 1u);
                const Matrix canonical_down = make_matrix(shape.ff, shape.hidden, bits, codebook, seed + 2u);
                seed += 3u;
                const std::vector<float> hidden_input = make_input(shape.hidden, seed);
                const std::vector<float> ff_input = make_input(shape.ff, seed + 1u);
                const std::string label = "matrix " + std::to_string(shape.hidden) + "x" + std::to_string(shape.ff) +
                                         " bits" + std::to_string(bits) + " cb" + std::to_string(codebook);
                for (const format::TileOrder order : kCpuOrders) {
                    const Matrix cpu_gate = reordered(canonical_gate, order);
                    const Matrix cpu_up = reordered(canonical_up, order);
                    const Matrix cpu_down = reordered(canonical_down, order);
                    const std::string tag = std::string(" [") + order_name(order) + "]";
                    check_matrix(canonical_gate, cpu_gate, order, hidden_input, label + " gate" + tag, projections,
                                 values);
                    check_matrix(canonical_up, cpu_up, order, hidden_input, label + " up" + tag, projections, values);
                    check_matrix(canonical_down, cpu_down, order, ff_input, label + " down" + tag, projections, values);

                    // The expert composition: gate, up, the product and down.
                    const format::ExpertView canonical_expert{
                        format::ProjectionView(view(canonical_gate, format::TileOrder::canonical)),
                        format::ProjectionView(view(canonical_up, format::TileOrder::canonical)),
                        format::ProjectionView(view(canonical_down, format::TileOrder::canonical))};
                    const format::ExpertView cpu_expert{format::ProjectionView(view(cpu_gate, order)),
                                                        format::ProjectionView(view(cpu_up, order)),
                                                        format::ProjectionView(view(cpu_down, order))};
                    const size_t max_k = static_cast<size_t>(std::max(shape.hidden, shape.ff));
                    std::vector<float> scratch_input(max_k);
                    std::vector<float> gate(static_cast<size_t>(shape.ff));
                    std::vector<float> up(static_cast<size_t>(shape.ff));
                    std::array<uint16_t, 256> tile{};
                    cpu_exl3::ExpertScratch canonical_scratch{{scratch_input.data(), scratch_input.size(), tile.data(),
                                                               tile.size()},
                                                              gate.data(), gate.size(), up.data(), up.size()};
                    std::vector<float> canonical_y(static_cast<size_t>(shape.hidden));
                    std::vector<float> cpu_y(static_cast<size_t>(shape.hidden));
                    const cpu_exl3::Status canonical_status =
                        cpu_exl3::expert(canonical_expert, hidden_input.data(), shape.hidden, canonical_y.data(),
                                         shape.hidden, 1, canonical_scratch);
                    const cpu_exl3::Status cpu_status = cpu_exl3::expert(cpu_expert, hidden_input.data(), shape.hidden,
                                                                         cpu_y.data(), shape.hidden, 1,
                                                                         canonical_scratch);
                    check(canonical_status == cpu_exl3::Status::ok && cpu_status == cpu_exl3::Status::ok &&
                              same_bits(canonical_y, cpu_y) && all_finite(cpu_y),
                          label + " expert is bit identical" + tag);
                    compared += canonical_y.size();
                }
            }
        }
    }

    // A mixed-up order must be visible: the same bytes read with the other
    // order cannot give the same answer, or the comparison above proves
    // nothing.  The two CPU orders are not the same bytes either, so one
    // cannot be read as the other.
    {
        const Matrix canonical = make_matrix(256, 256, 4, 2, 99u);
        std::vector<uint16_t> canonical_w(static_cast<size_t>(canonical.k * canonical.n));
        const format::MatrixView canonical_view = view(canonical, format::TileOrder::canonical);
        cpu_exl3::reconstruct(canonical_view, canonical_w.data(), canonical_w.size());
        for (const format::TileOrder order : kCpuOrders) {
            const Matrix cpu = reordered(canonical, order);
            std::vector<uint16_t> mixed_w(static_cast<size_t>(cpu.k * cpu.n));
            cpu_exl3::reconstruct(view(cpu, format::TileOrder::canonical), mixed_w.data(), mixed_w.size());
            check(canonical_w != mixed_w, std::string("a mixed-up ") + order_name(order) + " order changes the result");
            // The other CPU order's bytes read with this one are wrong too.
            for (const format::TileOrder other : kCpuOrders) {
                if (other == order) continue;
                const Matrix wrong = reordered(canonical, other);
                std::vector<uint16_t> wrong_w(static_cast<size_t>(wrong.k * wrong.n));
                cpu_exl3::reconstruct(view(wrong, order), wrong_w.data(), wrong_w.size());
                check(canonical_w != wrong_w, std::string("the ") + order_name(other) + " bytes read as " +
                                                   order_name(order) + " change the result");
            }
        }
    }

    // A non-finite or a tiny input sends the work back to the exact path, and
    // the exact path must read the same tiles.
    {
        const Matrix canonical = make_matrix(256, 256, 4, 2, 4242u);
        for (const format::TileOrder order : kCpuOrders) {
            const Matrix cpu = reordered(canonical, order);
            for (const float poison : {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::denorm_min()}) {
                std::vector<float> x = make_input(256, 11u);
                x[7] = poison;
                std::vector<float> canonical_input(256);
                std::array<uint16_t, 256> tile{};
                cpu_exl3::LinearScratch scratch{canonical_input.data(), canonical_input.size(), tile.data(), tile.size()};
                std::vector<float> canonical_y(256), cpu_y(256);
                const cpu_exl3::Status canonical_status =
                    cpu_exl3::linear(view(canonical, format::TileOrder::canonical), x.data(), canonical_y.data(), scratch);
                const cpu_exl3::Status cpu_status =
                    cpu_exl3::linear(view(cpu, order), x.data(), cpu_y.data(), scratch);
                check(canonical_status == cpu_exl3::Status::ok && cpu_status == cpu_exl3::Status::ok &&
                          same_bits(canonical_y, cpu_y),
                      std::string("the exact fallback reads the same tiles [") + order_name(order) + "]");
            }
        }
    }

    std::printf("exl3_cpu_layout_parity: %d failures, %zu packed projections, %zu expert output values, %zu matrix "
                "values compared (integer dot %s)\n",
                failures, projections, compared, values,
                cpu_exl3::mul1_intdot_enabled() ? "enabled" : "disabled");
    return failures == 0 ? 0 : 1;
}
