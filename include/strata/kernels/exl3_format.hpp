// SPDX-FileCopyrightText: 2025 Turboderp
// SPDX-FileCopyrightText: 2026 jyohukuchan
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// EXL3 packed tiles follow exllamav3's MIT-licensed trellis storage rules.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::exl3 {

/// The order of the compressed 16x16 tiles inside a packed projection's trellis.
///
/// `canonical` is the format's own order: the tile (tile_k, tile_n) sits at
/// `tile_k * n_tiles + tile_n`.  It is the only order the device kernels read.
///
/// `cpu_block` reorders whole tiles so that the tiles of one 128-output block
/// are adjacent: the tile (tile_k, tile_n) sits at
/// `(n_tile / B) * k_tiles * B + tile_k * B + n_tile % B` with B = 128 / 16.
/// The CPU walks the K tiles of one N tile with a stride of B tiles, so the
/// eight N tiles of a block are eight interleaved streams.
///
/// `cpu_nmajor` puts the K tiles of one N tile next to each other: the tile
/// (tile_k, tile_n) sits at `n_tile * k_tiles + tile_k`.  The CPU's walk of
/// the K tiles is then one sequential run, and the N tiles of a 128-output
/// block follow one another, so a task reads one contiguous stretch.
///
/// Only the order of whole tiles changes; the packed bits inside a tile, the
/// scales, the bias and the arithmetic do not.  The CPU reads the two CPU
/// orders, and a blob that carries one is identified by the descriptor, never
/// by guessing.
enum class TileOrder : uint8_t {
    canonical = 0,
    cpu_block = 1,
    cpu_nmajor = 2,
};

/// Whether the order is one of the CPU's own orders, and not the format's.
constexpr bool tile_order_is_cpu(TileOrder order) { return order != TileOrder::canonical; }

/// The number of 16-wide tiles one 128-output block covers.  128 outputs are
/// the unit of the H128 transform, so a task never covers less than one block.
/// This is the unit of the transform and of the task, not of the tile order:
/// the two are separate, and an order may group any number of tiles.
constexpr int64_t kTileBlock = 128 / 16;

/// The index of the tile (tile_k, tile_n) inside a trellis of the given order.
/// `tiles_k` and `tiles_n` are the tile counts of the projection.
constexpr size_t trellis_tile_index(TileOrder order, int64_t tile_k, int64_t tile_n, int64_t tiles_k,
                                    int64_t tiles_n) {
    if (order == TileOrder::cpu_block)
        return static_cast<size_t>((tile_n / kTileBlock) * tiles_k * kTileBlock + tile_k * kTileBlock +
                                    tile_n % kTileBlock);
    if (order == TileOrder::cpu_nmajor) return static_cast<size_t>(tile_n * tiles_k + tile_k);
    return static_cast<size_t>(tile_k * tiles_n + tile_n);
}

/// All half-valued buffers contain IEEE binary16 bits. The pointers may refer to
/// host memory (CPU routines) or device memory (SYCL routines), respectively.
/// Packed storage is [k / 16, n / 16, 16 * bits], in uint16_t elements.
struct MatrixView {
    const uint16_t* trellis = nullptr;
    const uint16_t* suh = nullptr;
    const uint16_t* svh = nullptr;
    const uint16_t* bias = nullptr;
    int64_t k = 0;
    int64_t n = 0;
    int bits = 0;
    int codebook = 0;  // 0: legacy/default, 1: mcg, 2: mul1
    /// The order of the compressed tiles in `trellis`.  The device kernels
    /// accept the canonical order only; the CPU kernels read either.
    TileOrder tile_order = TileOrder::canonical;
};

/// Matrix offset inside a 16x16 tile for one of its 256 trellis states.
constexpr int tile_element_for_state(int state) {
    const int lane = state / 8;
    const int part = state % 8;
    const int r = (lane % 4) * 2 + (part % 4) / 2 * 8 + part % 2;
    const int c = lane / 4 + part / 4 * 8;
    return r * 16 + c;
}

constexpr int tile_state_for_element(int element) {
    const int r = element / 16;
    const int c = element % 16;
    return ((c % 8) * 4 + (r % 8) / 2) * 8 + c / 8 * 4 + r / 8 * 2 + r % 2;
}

}  // namespace strata::kernels::exl3
