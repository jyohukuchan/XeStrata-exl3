// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include "strata/artifact/exl3.hpp"
#include "strata/artifact/exl3_model_map.hpp"
#include "strata/kernels/ple.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::core {

class Exl3Dense;
class Runtime;

/// Owns the EXL3 representation of the single Qwen PLE block.
///
/// The key and value projections remain separate Exl3Dense owners.  This is
/// deliberate: a raw F16/BF16/F32 projection keeps its source dtype and a
/// packed projection keeps its trellis representation until the caller runs
/// the preprojection.  `weights()` contains only the PLE post-op data (the
/// F32 norm vectors and the F16 convolution); its legacy key/value pointers
/// are left null for this owner and must not be interpreted as a conversion to
/// the native GGUF layout.
///
/// The object must outlive every captured graph using key(), value(), or
/// weights(). close() drains the selected runtime before releasing device
/// allocations. prepare_stream() must be called for every queue used by a
/// graph before capture; load() prepares the default compute queue.
class Exl3Ple {
public:
    Exl3Ple();
    ~Exl3Ple();
    Exl3Ple(const Exl3Ple &) = delete;
    Exl3Ple &operator=(const Exl3Ple &) = delete;

    /// Build the model map from the archive config and load the PLE layer.
    bool load(const strata::artifact::exl3::Archive &archive, int device, int64_t max_rows, std::string &err);

    /// Load using a map already built by the model loader.  The map supplies
    /// the source prefix; layer ids and PLE geometry are still checked against
    /// config.json here.
    bool load(const strata::artifact::exl3::Archive &archive, const strata::artifact::exl3::ModelMap &map, int device,
              int64_t max_rows, std::string &err);

    bool prepare_stream(void *stream, std::string &err);

    /// Synchronized teardown. The overload without an error string is useful
    /// from exception/destructor paths; errors are intentionally swallowed
    /// there after Runtime::finish_all has been attempted.
    bool close(std::string &err);
    void close() noexcept;

    const strata::kernels::PleWeights &weights() const { return weights_; }
    strata::kernels::PleWeights &weights() { return weights_; }
    Exl3Dense *key() const { return key_.get(); }
    Exl3Dense *value() const { return value_.get(); }
    /// Persistent single-token preprojection destinations. They are disjoint
    /// from the PLE scratch and remain valid until close().
    float *pre_key() const { return static_cast<float *>(pre_key_); }
    float *pre_value() const { return static_cast<float *>(pre_value_); }
    bool loaded() const { return runtime_ != nullptr && key_ != nullptr && value_ != nullptr; }
    int device() const { return device_; }

private:
    bool load_with_map(const strata::artifact::exl3::Archive &archive, const strata::artifact::exl3::ModelMap &map,
                       int device, int64_t max_rows, std::string &err);
    bool upload_f32(const float *data, size_t count, void *&destination, const char *label, std::string &err);
    bool upload_bytes(const std::vector<uint8_t> &data, void *&destination, const char *label, std::string &err);
    bool load_norm(const strata::artifact::exl3::Archive &archive, const std::string &name, void *&destination,
                   const char *label, std::string &err);
    bool load_projection(const strata::artifact::exl3::Archive &archive, const std::string &name, int64_t input_width,
                         int64_t output_width, int64_t max_rows, std::unique_ptr<Exl3Dense> &destination,
                         std::string &err);
    bool validate_config(const strata::artifact::exl3::Archive &archive, int &source_layer, std::string &err) const;
    bool release(std::string *err) noexcept;

    std::unique_ptr<Exl3Dense> key_;
    std::unique_ptr<Exl3Dense> value_;
    strata::kernels::PleWeights weights_{};
    void *norm_key_ = nullptr;
    void *norm_query_ = nullptr;
    void *norm_conv_ = nullptr;
    void *conv1d_f16_ = nullptr;
    void *pre_key_ = nullptr;
    void *pre_value_ = nullptr;
    Runtime *runtime_ = nullptr;
    int device_ = -1;
};

} // namespace strata::core
