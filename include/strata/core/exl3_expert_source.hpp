// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// A bounded EXL3 safetensors ExpertSource adapter.  Codec rules stay in the
// artifact and kernel components; this class owns only selected routed blobs.
#pragma once

#include "strata/artifact/exl3_expert.hpp"
#include "strata/core/expert_source.hpp"

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace strata::core {

/// ExpertSource for an explicit flat `(layer * n_expert + expert)` name table.
/// Metadata is discovered at open; payloads are materialized only for the
/// requested layer's unique ids in `begin_layer`.
class Exl3ExpertSource final : public ExpertSource {
public:
    using Names = std::vector<strata::artifact::exl3::ExpertTensorNames>;

    Exl3ExpertSource() = default;
    Exl3ExpertSource(std::shared_ptr<const strata::artifact::exl3::Archive> archive, int64_t n_layers,
                     int64_t n_expert, Names names);
    ~Exl3ExpertSource() override = default;
    Exl3ExpertSource(const Exl3ExpertSource&) = delete;
    Exl3ExpertSource& operator=(const Exl3ExpertSource&) = delete;

    bool open(std::shared_ptr<const strata::artifact::exl3::Archive> archive, int64_t n_layers, int64_t n_expert,
              Names names, std::string& err);
    bool open(const std::filesystem::path& directory, int64_t n_layers, int64_t n_expert, Names names,
              std::string& err);

    const uint8_t* blob(int64_t layer, int64_t expert) override;
    bool transient(int64_t layer, int64_t expert) const override;
    bool copy_blob(int64_t layer, int64_t expert, uint8_t* dst) override;
    const strata::kernels::exl3::ExpertBlobLayout* exl3_layout(int64_t layer, int64_t expert) const override;
    void begin_layer(int64_t layer, const int32_t* ids, int64_t k) override;
    int64_t reads() const override { return reads_.load(std::memory_order_relaxed); }

    const std::string& last_error() const { return failure_; }
    int64_t n_layers() const { return n_layers_; }
    int64_t n_expert() const { return n_expert_; }

private:
    bool valid_pair(int64_t layer, int64_t expert) const;
    bool ensure_materialized_locked(int64_t layer, int64_t expert);
    size_t index(int64_t layer, int64_t expert) const;

    std::shared_ptr<const strata::artifact::exl3::Archive> archive_;
    int64_t n_layers_ = 0;
    int64_t n_expert_ = 0;
    Names names_;
    std::vector<strata::kernels::exl3::ExpertBlobLayout> layouts_;
    std::vector<std::unique_ptr<strata::artifact::exl3::ExpertBlob>> materialized_;
    std::vector<size_t> active_indices_;
    mutable std::mutex mutex_;
    int64_t active_layer_ = -1;
    std::string failure_;
    std::atomic<int64_t> reads_{0};
};

}  // namespace strata::core
