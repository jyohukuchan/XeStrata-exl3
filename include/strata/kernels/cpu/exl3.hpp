// SPDX-FileCopyrightText: 2025 Turboderp
// SPDX-FileCopyrightText: 2026 jyohukuchan
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// CPU EXL3 trellis extraction, codebooks and the scalar/AVX2-floor linear path.
// The packed tile and codebook rules are derived from exllamav3's MIT-licensed
// implementation.  This interface deliberately keeps no ownership of weights.
#pragma once

#include "strata/kernels/exl3_format.hpp"

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu::exl3 {

/// All routines reject malformed descriptors instead of interpreting them as
/// another weight format.  `scratch_too_small` is returned before any output
/// is written by the linear routine.
enum class Status {
    ok = 0,
    invalid_argument,
    unsupported_bits,
    unsupported_codebook,
    unsupported_layout,
    missing_data,
    output_too_small,
    scratch_too_small,
    size_overflow,
};

const char* status_string(Status status);

/// Validate the packed trellis shape and its required data pointers.  Bits
/// 1..8 are accepted: 2..5 are the primary EXL3 targets and 1/6/7/8 are kept
/// for valid checkpoint head and MTP exceptions.
Status validate(const strata::kernels::exl3::MatrixView& matrix);

/// Extract the 256 16-bit tail-biting states from one 16x16 packed tile.
/// `packed` contains exactly 16 * bits uint16 words and `states` has room for
/// at least 256 entries.  States are returned in trellis order, before the
/// tensor-core permutation.
Status unpack_trellis_states(const uint16_t* packed, int bits, uint16_t* states, size_t state_count);

/// Decode state words with EXL3's legacy/default (0), MCG (1), or MUL1 (2)
/// codebook.  Results are IEEE binary16 bit patterns, including the half
/// rounding at each codebook operation boundary.
Status decode_codebook(const uint16_t* states, size_t state_count, int codebook, uint16_t* values,
                       size_t value_count);

/// Decode one tile into row-major 16x16 binary16 values.  The tensor-core
/// permutation is applied, while `suh`/`svh` are intentionally not applied:
/// the result is W-hat in EXL3's transformed basis.
Status decode_tile(const strata::kernels::exl3::MatrixView& matrix, int64_t tile_k, int64_t tile_n,
                   uint16_t* values, size_t value_count);

/// Reconstruct W-hat into caller-owned row-major binary16 storage.  This is an
/// explicit diagnostic/reference operation.  The per-token linear path below
/// decodes one tile at a time and never allocates this k*n matrix.
Status reconstruct(const strata::kernels::exl3::MatrixView& matrix, uint16_t* values, size_t value_count);

/// Caller-owned bounded workspace for `linear`.  `input_basis` needs k float
/// entries and `tile` needs 256 uint16 entries.  The caller may reuse it for
/// every token and matrix.  The buffers may alias neither one another nor y.
struct Mul1Input;

struct LinearScratch {
    float* input_basis = nullptr;
    size_t input_count = 0;
    uint16_t* tile = nullptr;
    size_t tile_count = 0;
    // Optional experimental Q8 workspace; absent means exact F16/FP64 path.
    int32_t* mul1_duplicated = nullptr;
    size_t mul1_count = 0;
    const Mul1Input* mul1_prepared = nullptr;
};

/// Number of bytes needed by a conventional contiguous scratch allocation:
/// k FP32 input-basis values followed by 256 uint16 tile values.  Returns zero
/// for a malformed descriptor or an overflowing size.
size_t scratch_bytes(const strata::kernels::exl3::MatrixView& matrix);

/// Batch-1 linear operation for W-hat [k,n]:
///
///   xh = H128(suh * x) / sqrt(128)
///   z[n] = sum_k float64(W-hat[k,n]) * xh[k], rounded to FP32
///   y  = H128(z) / sqrt(128) * svh + bias
///
/// Hadamard butterflies and scale/bias operations are FP32.  W-hat values
/// are converted from their exact decoded binary16 bits.  k and n must be
/// positive multiples of 128 because the EXL3 transforms are H128 blocks.
/// The explicitly enabled integer prototype uses Q8 input and pre-RNE affine
/// CB2 weights only when optional MUL1 scratch is supplied; otherwise exact.
Status linear(const strata::kernels::exl3::MatrixView& matrix, const float* x, float* y, LinearScratch& scratch);

/// Prepare one packed projection's H128 input basis without computing a dot.  The arithmetic order matches linear().
Status prepare_input_basis(const strata::kernels::exl3::MatrixView& matrix, const float* x, float* basis);

/// Compute one 128-column packed output block from a prepared input basis.  The block includes Hpost, svh and bias.
/// `col0` must be 128-aligned; only this block is written.  The scratch input vector is used as a local dot block.
Status linear_output_block(const strata::kernels::exl3::MatrixView& matrix, const float* input_basis, int64_t col0,
                           float* y, LinearScratch& scratch);

/// Exact scalar SiLU and multiply used by the existing routed expert composition.
void silu_mul_block(const float* gate, const float* up, float* out, size_t count);

/// Convenience wrapper using a bounded temporary input vector and one tile;
/// it still never materialises the full weight matrix.  Callers on a token
/// hot path should provide LinearScratch to avoid the input allocation.
Status linear(const strata::kernels::exl3::MatrixView& matrix, const float* x, float* y);

}  // namespace strata::kernels::cpu::exl3
