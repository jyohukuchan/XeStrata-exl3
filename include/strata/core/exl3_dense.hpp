// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include "strata/kernels/exl3_format.hpp"
#include "strata/kernels/raw_linear_format.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace strata::artifact::exl3 {
class Archive;
}  // namespace strata::artifact::exl3

namespace strata::core {
class Runtime;
class WeightTable;
struct WeightRef;

/// Owns one explicitly mapped EXL3 dense matrix and its graph-stable scratch.
/// The owner must outlive every captured graph that calls `run`; its attached
/// WeightTable must also outlive the owner so the optional reference can be
/// detached safely during destruction.
class Exl3Dense {
public:
    Exl3Dense();
    ~Exl3Dense();
    Exl3Dense(const Exl3Dense&) = delete;
    Exl3Dense& operator=(const Exl3Dense&) = delete;

    /// Loads `hf_name` and attaches it to the canonical `output.weight` row.
    /// The initial head path runs one row at a time; `max_rows` keeps the
    /// persistent scratch contract explicit for a later batched head.
    bool load(const std::filesystem::path& directory, const std::string& hf_name, WeightTable& table,
              std::string& err, int64_t max_rows = 1);
    /// Reuses an already validated archive.  No archive-wide host metadata is
    /// retained after upload; only the selected matrix components are owned.
    bool load(const strata::artifact::exl3::Archive& archive, const std::string& hf_name, WeightTable& table,
              std::string& err, int64_t max_rows = 1);

    /// Prepare one runtime queue's persistent scratch before graph capture.
    /// The default compute queue is prepared by `load`; callers must prepare
    /// every other queue before its first capture.  Each explicit registration
    /// allocates a workspace sized from this matrix and `max_rows`.
    bool prepare_stream(void* stream, std::string& err);

    /// Runs the packed or raw matrix with FP32 input/output and caller-owned strides.
    /// The stream must have been registered with `prepare_stream` first.
    bool run(const float* x, int64_t ldx, float* y, int64_t ldy, int64_t rows, void* stream, std::string& err);

    bool raw() const { return raw_; }
    strata::kernels::raw::DType raw_dtype() const { return raw_view_.dtype; }

private:
    struct StreamScratch {
        const void* queue = nullptr;
        void* data = nullptr;
    };

    void reset();
    bool pointers_owned_by(const Runtime* runtime) const;

    strata::kernels::exl3::MatrixView view_;
    strata::kernels::raw::LinearView raw_view_;
    void* trellis_ = nullptr;
    void* suh_ = nullptr;  // codespell:ignore suh
    void* svh_ = nullptr;
    void* bias_ = nullptr;
    void* raw_weights_ = nullptr;
    void* raw_bias_ = nullptr;
    size_t scratch_elements_ = 0;
    int64_t max_rows_ = 0;
    Runtime* runtime_ = nullptr;
    bool raw_ = false;
    std::vector<StreamScratch> streams_;
    WeightRef* attached_ = nullptr;
};

}  // namespace strata::core
