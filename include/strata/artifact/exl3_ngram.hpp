// SPDX-FileCopyrightText: 2025 Turboderp
// SPDX-FileCopyrightText: 2026 jyohukuchan
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// CPU EXL3 n-gram safetensors sidecar reader.
#pragma once

#include "strata/artifact/exl3.hpp"
#include "strata/kernels/cpu/exl3_ngram.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace strata::artifact::exl3_ngram {

/// A blocking reader over one validated exl3_ngram_trellis sidecar.  It reads
/// only requested rows; the complete table remains file-backed.
class Reader {
public:
    explicit Reader(const std::filesystem::path& path);
    ~Reader();
    Reader(Reader&&) noexcept;
    Reader& operator=(Reader&&) noexcept;
    Reader(const Reader&) = delete;
    Reader& operator=(const Reader&) = delete;

    const std::filesystem::path& path() const;
    const std::string& prefix() const;
    int bits() const;
    size_t words_per_row() const;
    int64_t row_count() const;
    int64_t rows_per_shard() const;
    int64_t num_heads() const;
    const std::vector<int64_t>& head_offsets() const;
    const std::vector<int64_t>& head_vocab_sizes() const;
    const std::vector<int64_t>& layer_multipliers() const;

    /// Read requested global rows in caller order into packed row-major
    /// uint16 storage.  A null row_ids pointer means 0..count-1.
    void read_rows(const int64_t* row_ids, size_t count, std::vector<uint16_t>& packed) const;

    /// Decode requested rows to FP32, using explicit host head IDs when given.
    /// Without explicit heads, every row must fall inside one of the declared
    /// non-overlapping head ranges; gap and trailing padding rows are rejected.
    strata::kernels::cpu::exl3::Status decode_rows(
        const int64_t* row_ids, const int32_t* explicit_head_ids, size_t count, float* output,
        size_t output_stride) const;

    /// Reuse the sidecar's signed-int64 wrap, EOS and modulo rules to produce
    /// global row IDs in [output_rows, num_heads] row-major storage.
    void lookup_ids(const int64_t* tokens, size_t token_count, int ngram_size, int heads_per_ngram,
                    int64_t eos_token_id, std::vector<int64_t>& output) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata::artifact::exl3_ngram
