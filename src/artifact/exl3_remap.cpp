// SPDX-FileCopyrightText: 2025 Turboderp
// SPDX-FileCopyrightText: 2026 jyohukuchan
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// Tile storage follows exllamav3's MIT-licensed packed representation.
#include "strata/artifact/exl3_remap.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <numeric>
#include <utility>

namespace strata::artifact::exl3 {
namespace {

bool block_map(const std::vector<int64_t>& requested, size_t count, bool subset,
                std::vector<int64_t>& result, std::string& error) {
    if (requested.empty()) {
        result.resize(count);
        std::iota(result.begin(), result.end(), int64_t{0});
        return true;
    }
    if ((!subset && requested.size() != count) || requested.size() > count) {
        error = "EXL3 remap: block map has the wrong length";
        return false;
    }
    std::vector<uint8_t> seen(count, 0);
    for (int64_t index : requested) {
        if (index < 0 || (uint64_t) index >= count || seen[(size_t) index] != 0) {
            error = "EXL3 remap: block indices must be unique and in range";
            return false;
        }
        seen[(size_t) index] = 1;
    }
    result = requested;
    return true;
}

void reorder_values(const std::vector<uint16_t>& source, const std::vector<int64_t>& blocks,
                     std::vector<uint16_t>& destination) {
    if (source.empty()) return;
    destination.resize(blocks.size() * 128);
    for (size_t block = 0; block < blocks.size(); ++block)
        std::copy_n(source.data() + (size_t) blocks[block] * 128, 128, destination.data() + block * 128);
}

}  // namespace

bool remap_matrix_blocks(const MaterializedMatrix& source, const std::vector<int64_t>& input_blocks,
                         const std::vector<int64_t>& output_blocks, MaterializedMatrix& destination,
                         std::string& error) {
    error.clear();
    if (source.k <= 0 || source.n <= 0 || source.k % 128 != 0 || source.n % 128 != 0 ||
        source.bits < 1 || source.bits > 8 || source.codebook < 0 || source.codebook > 2) {
        error = "EXL3 remap: invalid H128 matrix geometry or codec";
        return false;
    }
    const size_t kt = (size_t) (source.k / 16), nt = (size_t) (source.n / 16);
    const size_t tile_words = (size_t) source.bits * 16;
    const size_t max_words = (size_t) std::numeric_limits<ptrdiff_t>::max() / sizeof(uint16_t);
    if (kt > max_words / nt || kt * nt > max_words / tile_words ||
        source.trellis.size() != kt * nt * tile_words || source.suh.size() != (uint64_t) source.k ||
        source.svh.size() != (uint64_t) source.n ||
        (!source.bias.empty() && source.bias.size() != (uint64_t) source.n)) {
        error = "EXL3 remap: component lengths do not match the matrix";
        return false;
    }
    std::vector<int64_t> inputs, outputs;
    if (!block_map(input_blocks, (size_t) (source.k / 128), false, inputs, error) ||
        !block_map(output_blocks, (size_t) (source.n / 128), true, outputs, error)) return false;
    MaterializedMatrix result;
    result.k = source.k;
    result.n = (int64_t) outputs.size() * 128;
    result.bits = source.bits;
    result.codebook = source.codebook;
    const size_t out_nt = outputs.size() * 8;
    result.trellis.resize(kt * out_nt * tile_words);
    for (size_t row = 0; row < kt; ++row) {
        const size_t source_row = (size_t) inputs[row / 8] * 8 + row % 8;
        for (size_t col = 0; col < out_nt; ++col) {
            const size_t source_col = (size_t) outputs[col / 8] * 8 + col % 8;
            std::copy_n(source.trellis.data() + (source_row * nt + source_col) * tile_words, tile_words,
                        result.trellis.data() + (row * out_nt + col) * tile_words);
        }
    }
    reorder_values(source.suh, inputs, result.suh);
    reorder_values(source.svh, outputs, result.svh);
    reorder_values(source.bias, outputs, result.bias);
    destination = std::move(result);
    return true;
}

}  // namespace strata::artifact::exl3
