// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "strata/core/exl3_expert_source.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace strata::core {

Exl3ExpertSource::Exl3ExpertSource(std::shared_ptr<const strata::artifact::exl3::Archive> archive,
                                   int64_t n_layers, int64_t n_expert, Names names) {
    std::string err;
    if (!open(std::move(archive), n_layers, n_expert, std::move(names), err)) failure_ = std::move(err);
}

size_t Exl3ExpertSource::index(int64_t layer, int64_t expert) const {
    return static_cast<size_t>(layer) * static_cast<size_t>(n_expert_) + static_cast<size_t>(expert);
}

bool Exl3ExpertSource::valid_pair(int64_t layer, int64_t expert) const {
    return archive_ != nullptr && layer >= 0 && layer < n_layers_ && expert >= 0 && expert < n_expert_;
}

bool Exl3ExpertSource::open(std::shared_ptr<const strata::artifact::exl3::Archive> archive, int64_t n_layers,
                            int64_t n_expert, Names names, std::string& err) {
    std::lock_guard<std::mutex> lock(mutex_);
    err.clear();
    archive_.reset();
    names_.clear();
    layouts_.clear();
    materialized_.clear();
    active_indices_.clear();
    active_layer_ = -1;
    failure_.clear();
    n_layers_ = n_expert_ = 0;
    reads_.store(0, std::memory_order_relaxed);
    if (archive == nullptr || n_layers <= 0 || n_expert <= 0) {
        err = "exl3 source: archive and positive dimensions are required";
        failure_ = err;
        return false;
    }
    if (static_cast<uint64_t>(n_layers) > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(n_expert)) {
        err = "exl3 source: layer/expert dimensions overflow";
        failure_ = err;
        return false;
    }
    const uint64_t count = static_cast<uint64_t>(n_layers) * static_cast<uint64_t>(n_expert);
    if (count > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) || names.size() != static_cast<size_t>(count)) {
        err = "exl3 source: flat name table size does not match layer/expert dimensions";
        failure_ = err;
        return false;
    }
    archive_ = std::move(archive);
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    names_ = std::move(names);
    layouts_.resize(names_.size());
    materialized_.resize(names_.size());
    for (size_t i = 0; i < names_.size(); ++i) {
        std::string why;
        if (!strata::artifact::exl3::describe_expert(*archive_, names_[i], layouts_[i], why)) {
            err = "exl3 source: expert " + std::to_string(i) + ": " + why;
            failure_ = err;
            archive_.reset(); names_.clear(); layouts_.clear(); materialized_.clear(); active_indices_.clear();
            n_layers_ = n_expert_ = 0;
            return false;
        }
    }
    for (int64_t layer = 0; layer < n_layers; ++layer) {
        int64_t hidden = -1;
        for (int64_t expert = 0; expert < n_expert; ++expert) {
            const auto& layout = layouts_[index(layer, expert)];
            if (hidden < 0) hidden = layout.gate.k;
            if (layout.down.n != hidden) {
                err = "exl3 source: experts in one layer have different hidden dimensions";
                failure_ = err;
                archive_.reset(); names_.clear(); layouts_.clear(); materialized_.clear(); active_indices_.clear();
                n_layers_ = n_expert_ = 0;
                return false;
            }
        }
    }
    return true;
}

bool Exl3ExpertSource::open(const std::filesystem::path& directory, int64_t n_layers, int64_t n_expert,
                            Names names, std::string& err) {
    try {
        auto archive = std::make_shared<strata::artifact::exl3::Archive>(directory);
        return open(std::move(archive), n_layers, n_expert, std::move(names), err);
    } catch (const std::exception& e) {
        err = std::string("exl3 source: ") + e.what();
        failure_ = err;
        return false;
    }
}

bool Exl3ExpertSource::ensure_materialized_locked(int64_t layer, int64_t expert) {
    if (!valid_pair(layer, expert)) {
        failure_ = "exl3 source: layer or expert is out of range";
        return false;
    }
    const size_t at = index(layer, expert);
    if (materialized_[at] != nullptr) return true;
    auto blob = std::make_unique<strata::artifact::exl3::ExpertBlob>();
    std::string err;
    if (!strata::artifact::exl3::materialize_expert(*archive_, names_[at], *blob, err)) {
        failure_ = "exl3 source: materializing layer " + std::to_string(layer) + " expert " +
                   std::to_string(expert) + ": " + err;
        return false;
    }
    if (blob->layout.total_bytes != layouts_[at].total_bytes) {
        failure_ = "exl3 source: materialized blob layout changed after metadata discovery";
        return false;
    }
    materialized_[at] = std::move(blob);
    active_indices_.push_back(at);
    return true;
}

void Exl3ExpertSource::begin_layer(int64_t layer, const int32_t* ids, int64_t k) {
    std::lock_guard<std::mutex> lock(mutex_);
    failure_.clear();
    for (const size_t at : active_indices_) materialized_[at].reset();
    active_indices_.clear();
    active_layer_ = -1;
    if (!valid_pair(layer, 0) || (ids == nullptr && k != 0) || k < 0) {
        failure_ = "exl3 source: invalid begin_layer request";
        return;
    }
    for (int64_t i = 0; i < k; ++i) {
        if (ids[i] < 0 || ids[i] >= n_expert_) {
            failure_ = "exl3 source: routed expert id is out of range";
            return;
        }
    }
    active_layer_ = layer;
    for (int64_t i = 0; i < k; ++i) {
        const int64_t expert = ids[i];
        if (!ensure_materialized_locked(layer, expert)) break;
    }
    if (!failure_.empty()) {
        for (const size_t at : active_indices_) materialized_[at].reset();
        active_indices_.clear();
        active_layer_ = -1;
    }
}

const uint8_t* Exl3ExpertSource::blob(int64_t layer, int64_t expert) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_layer_ != layer) {
        failure_ = "exl3 source: begin_layer must precede blob requests for a new layer";
        return nullptr;
    }
    if (!ensure_materialized_locked(layer, expert)) return nullptr;
    ++reads_;
    return materialized_[index(layer, expert)]->data();
}

bool Exl3ExpertSource::transient(int64_t layer, int64_t expert) const {
    return valid_pair(layer, expert);
}

bool Exl3ExpertSource::copy_blob(int64_t layer, int64_t expert, uint8_t* dst) {
    if (dst == nullptr || !valid_pair(layer, expert)) return false;
    std::shared_ptr<const strata::artifact::exl3::Archive> archive;
    strata::artifact::exl3::ExpertTensorNames names;
    strata::kernels::exl3::ExpertBlobLayout layout;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        archive = archive_;
        names = names_[index(layer, expert)];
        layout = layouts_[index(layer, expert)];
    }
    strata::artifact::exl3::ExpertBlob local;
    std::string err;
    if (!strata::artifact::exl3::materialize_expert(*archive, names, local, err) ||
        local.layout.total_bytes != layout.total_bytes)
        return false;
    std::memcpy(dst, local.data(), local.bytes());
    return true;
}

const strata::kernels::exl3::ExpertBlobLayout* Exl3ExpertSource::exl3_layout(int64_t layer, int64_t expert) const {
    if (!valid_pair(layer, expert)) return nullptr;
    return &layouts_[index(layer, expert)];
}

}  // namespace strata::core
