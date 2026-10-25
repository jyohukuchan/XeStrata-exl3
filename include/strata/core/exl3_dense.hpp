// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include "strata/kernels/exl3.hpp"
#include "strata/kernels/exl3_format.hpp"
#include "strata/kernels/raw_linear_format.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace strata::artifact::exl3 {
class Archive;
struct RawTensor;
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
    /// Loads `hf_name` and attaches it to the explicitly selected canonical row.
    /// The explicit form is used by non-head projections; callers own the name
    /// mapping and must keep the selected WeightTable row alive until destruction.
    bool load(const std::filesystem::path& directory, const std::string& hf_name, const std::string& canonical_name,
              WeightTable& table, std::string& err, int64_t max_rows = 1);
    /// Reuses an already validated archive.  No archive-wide host metadata is
    /// retained after upload; only the selected matrix components are owned.
    bool load(const strata::artifact::exl3::Archive& archive, const std::string& hf_name, WeightTable& table,
              std::string& err, int64_t max_rows = 1);
    /// Reuses an already validated archive and attaches to `canonical_name`.
    bool load(const strata::artifact::exl3::Archive& archive, const std::string& hf_name,
              const std::string& canonical_name, WeightTable& table, std::string& err, int64_t max_rows = 1);
    /// Reorder whole H128 blocks and optionally select output blocks before
    /// uploading packed weights. This preserves their exact quantization and
    /// supports explicit projection splits. Empty maps are identity; raw
    /// matrices require a separate raw-axis mapping and reject nonempty maps.
    bool load_remapped(const strata::artifact::exl3::Archive& archive, const std::string& hf_name,
                       const std::string& canonical_name, const std::vector<int64_t>& input_blocks,
                       const std::vector<int64_t>& output_blocks, WeightTable& table, std::string& err,
                       int64_t max_rows = 1);

    /// Own a projection without attaching to an external canonical WeightTable.
    /// The explicit target dimensions and block maps are validated identically
    /// to load_remapped(); no synthetic pack or GGUF weight is required.
    bool load_detached(const strata::artifact::exl3::Archive& archive, const std::string& hf_name,
                       int64_t input_width, int64_t output_width, const std::vector<int64_t>& input_blocks,
                       const std::vector<int64_t>& output_blocks, std::string& err, int64_t max_rows = 1);

    /// Upload an explicitly materialized raw [output,input] matrix after axis
    /// mapping. Weight dtype is preserved; an optional bias is widened to F32.
    bool load_raw_detached(const strata::artifact::exl3::RawTensor& weights,
                           const strata::artifact::exl3::RawTensor* bias, std::string& err, int64_t max_rows = 1);

    /// Prepare one runtime queue's persistent scratch before graph capture.
    /// The default compute queue is prepared by `load`; callers must prepare
    /// every other queue before its first capture.  Each explicit registration
    /// allocates a workspace sized from this matrix and `max_rows`.
    bool prepare_stream(void* stream, std::string& err);

    /// Runs the packed or raw matrix with FP32 input/output and caller-owned strides.
    /// The stream must have been registered with `prepare_stream` first.
    bool run(const float* x, int64_t ldx, float* y, int64_t ldy, int64_t rows, void* stream, std::string& err);

    /// Runs like `run`, with the caller's basis options.  The fused transform tails
    /// (strata/kernels/exl3_fused.hpp) own the stages a call disables, so the values are `run`'s
    /// when the same stages are applied around it.  A raw owner carries no basis stages and runs
    /// exactly as `run` does; the tails only take packed views (`packed`).
    bool run(const float* x, int64_t ldx, float* y, int64_t ldy, int64_t rows, void* stream,
             const strata::kernels::exl3::GemvOptions& options, std::string& err);

    /// The packed matrix view the fused transform tails read their `suh`/`svh`/bias from;
    /// null for a raw owner.
    const strata::kernels::exl3::MatrixView* packed() const { return raw_ ? nullptr : &view_; }

    /// The plane `run` transforms the input into on `stream` before the dot: the input-basis image
    /// a dot reads with `GemvOptions::input_hadamard` off.  `exl3_input_hadamard_dual_fp32` writes
    /// two owners' planes in one launch, which is what exposes this.  `rows` rows of the input
    /// width; false for a raw owner, one with no `suh`, an unprepared stream or a row count above
    /// the owner's `max_rows`.
    bool input_scratch(void* stream, int64_t rows, float*& plane, std::string& err);

    bool raw() const { return raw_; }
    strata::kernels::raw::DType raw_dtype() const { return raw_view_.dtype; }

    /// The FP16 GEMM prefill product of this projection: the same transform stages as
    /// `run` (input/output Hadamard and bias, per `options`), with the dot itself run as an FP16
    /// product (strata/prefill/gemm.hpp) over weights dequantized once by `exl3_reconstruct` and
    /// cached transposed as W[N, K].  A raw owner takes the same product over its own weights:
    /// F16 weights are already W[N, K] and are used in place (no cache is charged), BF16/F32
    /// weights are converted once into the cache.  False (with `err`) when this owner cannot
    /// take the path (a bias the output transform would not add, unaligned strides, or no room
    /// in the shared cache budget yet); the caller then keeps `run`.  A failure is not latched:
    /// the call is retried after a bounded number of fallbacks, so room freed later in the
    /// process (a smaller resident set, another owner's reset) lets the path take over.
    /// Requires tight strides (`ldx == k`, `ldy == n`).
    bool run_fp16_gemm(const float* x, int64_t ldx, float* y, int64_t ldy, int64_t rows, void* stream,
                       const strata::kernels::exl3::GemvOptions& options, std::string& err);
    /// Whether the FP16 weights are already cached (the one-time dequant has run).
    bool fp16_ready() const { return fp16_w_ != nullptr; }

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
    void* fp16_w_ = nullptr;    ///< cached dequantized weights, FP16 W[N, K] (run_fp16_gemm)
    bool fp16_w_borrowed_ = false;  ///< fp16_w_ aliases raw_weights_ (raw F16); not owned, not charged
    size_t fp16_w_bytes_ = 0;   ///< bytes charged to the shared FP16 cache budget (0 when borrowed)
    void* fp16_x_ = nullptr;    ///< FP16 activations, fp16_x_elems_ halves
    size_t fp16_x_elems_ = 0;
    float* fp16_xh_ = nullptr;  ///< FP32 transform/product rows, fp16_xh_elems_ floats
    size_t fp16_xh_elems_ = 0;
    int fp16_backoff_ = 0;  ///< fallbacks to serve before the cache build is retried after a failure
    size_t scratch_elements_ = 0;
    int64_t max_rows_ = 0;
    Runtime* runtime_ = nullptr;
    bool raw_ = false;
    std::vector<StreamScratch> streams_;
    WeightRef* attached_ = nullptr;
    std::unique_ptr<WeightTable> detached_table_;
};

/// The shared byte budget of every owner's FP16 weight cache (`run_fp16_gemm`), and the room a
/// resident-expert cache must leave for it.  `STRATA_EXL3_PREFILL_FP16_CACHE_MIB` sets it in MiB
/// (0: no cache; the raw F16 in-place product is unaffected); unset, it is a share of what the
/// device reports: total/8, clamped to [256 MiB, 8 GiB] - measured against the dense FP16 set of
/// the Qwen3.8 trunk on a 32 GiB card, with the fallback keeping every other size correct.
uint64_t prefill_fp16_cache_budget_bytes(uint64_t total_bytes);

}  // namespace strata::core
