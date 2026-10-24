// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Per-device EXL3 dense and auxiliary-weight ownership.
#pragma once

#include "strata/artifact/exl3_model_map.hpp"
#include "strata/core/exl3_dense.hpp"
#include "strata/core/weights.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::core {

/// The first full-archive bridge owns dense matrices and the F32 auxiliary
/// vectors required by the existing layer/verify paths. Routed experts,
/// token embedding and MTP are deliberately separate owners for now.
struct Exl3WeightSetOptions {
    int device = -1;             ///< -1: the caller's current device
    int64_t layer_begin = 0;     ///< inclusive trunk layer
    int64_t layer_end = -1;      ///< exclusive; -1: all trunk layers
    bool include_head = false;   ///< output.weight and a qwen35 final norm
    bool include_final_hc = false;  ///< qwen3.8 combine-less output mixer
    bool include_mtp_block = false;  ///< mtp_fc plus layer<0 globals: the Qwen3.8 one-layer MTP map only
    int64_t max_rows = 1;        ///< persistent EXL3 scratch/capture row capacity
};

/// Owns one device's selected EXL3 dense matrices, raw auxiliary F32 buffers,
/// and the non-owning WeightRef registrations that expose them to the engine.
/// The WeightTable is borrowed and must outlive this object. Destruction first
/// restores or erases only this object's registrations, then destroys device
/// allocations and owners, switching to this object's device internally.
class Exl3WeightSet {
public:
    Exl3WeightSet() = default;
    ~Exl3WeightSet();
    Exl3WeightSet(const Exl3WeightSet&) = delete;
    Exl3WeightSet& operator=(const Exl3WeightSet&) = delete;

    /// Attach the selected trunk range from an already inspected archive.
    /// This does not call WeightTable::load and does not create a synthetic
    /// pack. Unsupported unrelated sidecars (for example PLE) do not prevent
    /// dense attachment, but a selected mapping that is missing, ambiguous,
    /// or incompatible fails before it is exposed.
    bool load(const strata::artifact::exl3::Archive& archive,
              const strata::artifact::exl3::ModelMap& map,
              WeightTable& table,
              const Exl3WeightSetOptions& options,
              std::string& err);

    /// Register every owner on an ordered queue before graph capture. A null
    /// stream means the selected device's compute queue.
    bool prepare_stream(void* stream, std::string& err);
    void close() noexcept;

    int device() const { return device_; }
    bool loaded() const { return table_ != nullptr; }
    size_t dense_count() const { return dense_.size(); }
    size_t auxiliary_count() const { return auxiliary_.size(); }

private:
    struct Registration {
        std::string name;
        bool inserted = false;
        WeightRef before;
        Exl3Dense* owner = nullptr;
        const void* data = nullptr;
    };

    bool attach_mapping(const strata::artifact::exl3::Archive& archive,
                        const strata::artifact::exl3::ProjectionMapping& mapping,
                        int64_t max_rows, std::string& err);
    bool attach_final_hc(const strata::artifact::exl3::Archive& archive,
                         const strata::artifact::exl3::ModelMap& map,
                         int64_t max_rows, std::string& err);
    bool register_ref(const std::string& name, const WeightRef& value, Exl3Dense* owner,
                      const void* data, std::string& err);
    bool register_matrix(const std::string& canonical, Exl3Dense& owner,
                         int64_t input, int64_t output, uint64_t bytes, std::string& err);
    bool register_aux(const std::string& canonical, const float* data,
                      int64_t elements, int64_t ne0, int64_t ne1,
                      uint64_t bytes, std::string& err);
    void reset_rows() noexcept;
    void reset_owners() noexcept;

    int device_ = -1;
    WeightTable* table_ = nullptr;
    std::vector<std::unique_ptr<Exl3Dense>> dense_;
    std::vector<void*> auxiliary_;
    std::vector<Registration> registrations_;
    std::vector<void*> prepared_streams_;
};

}  // namespace strata::core
