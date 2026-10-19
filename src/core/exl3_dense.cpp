// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "strata/core/exl3_dense.hpp"

#include "strata/artifact/exl3.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/runtime.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/exl3.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::core {
namespace {
void free_device(void*& pointer) {
    if (pointer != nullptr) {
        strata::gpu::free(pointer);
        pointer = nullptr;
    }
}

bool copy_component(void*& destination, const std::vector<uint16_t>& source, const char* label, std::string& err) {
    if (source.empty()) return true;
    if (source.size() > std::numeric_limits<size_t>::max() / sizeof(uint16_t)) {
        err = std::string("exl3 dense: ") + label + " size overflows size_t";
        return false;
    }
    const size_t bytes = source.size() * sizeof(uint16_t);
    destination = strata::gpu::alloc_device(bytes);
    if (destination == nullptr || !strata::gpu::copy(destination, source.data(), bytes)) {
        if (destination != nullptr) strata::gpu::free(destination);
        destination = nullptr;
        err = std::string("exl3 dense: uploading ") + label + " failed: " + strata::gpu::last_error();
        return false;
    }
    return true;
}

}  // namespace

Exl3Dense::Exl3Dense() = default;

Exl3Dense::~Exl3Dense() {
    if (attached_ != nullptr && attached_->exl3 == this) attached_->exl3 = nullptr;
    reset();
}

void Exl3Dense::reset() {
    free_device(trellis_);
    free_device(suh_);  // codespell:ignore suh
    free_device(svh_);
    free_device(bias_);
    for (StreamScratch& stream : streams_) free_device(stream.data);
    streams_.clear();
    view_ = {};
    scratch_elements_ = 0;
    max_rows_ = 0;
    runtime_ = nullptr;
    attached_ = nullptr;
}

bool Exl3Dense::pointers_owned_by(const Runtime* runtime) const {
    if (runtime == nullptr) return false;
    const void* pointers[] = {view_.trellis, view_.suh, view_.svh, view_.bias};
    for (const void* pointer : pointers)
        if (pointer != nullptr && Runtime::owner(pointer) != runtime) return false;
    for (const StreamScratch& stream : streams_)
        if (stream.data != nullptr && Runtime::owner(stream.data) != runtime) return false;
    return true;
}

bool Exl3Dense::load(const std::filesystem::path& directory, const std::string& hf_name, WeightTable& table,
                     std::string& err, int64_t max_rows) {
    if (runtime_ != nullptr || attached_ != nullptr) {
        err = "exl3 dense: the owner is already loaded";
        return false;
    }
    try {
        const strata::artifact::exl3::Archive archive(directory);
        return load(archive, hf_name, table, err, max_rows);
    } catch (const std::exception& error) {
        err = std::string("exl3 dense: ") + error.what();
        return false;
    }
}

bool Exl3Dense::load(const strata::artifact::exl3::Archive& archive, const std::string& hf_name, WeightTable& table,
                     std::string& err, int64_t max_rows) {
    err.clear();
    if (runtime_ != nullptr || attached_ != nullptr) {
        err = "exl3 dense: the owner is already loaded";
        return false;
    }
    if (hf_name.empty() || max_rows <= 0) {
        err = "exl3 dense: HF name and max_rows are required";
        return false;
    }
    try {
        const strata::artifact::exl3::LinearDescriptor* descriptor = nullptr;
        for (const auto& candidate : archive.linears()) {
            if (candidate.name == hf_name) {
                descriptor = &candidate;
                break;
            }
        }
        if (descriptor == nullptr) throw std::runtime_error("linear is missing: " + hf_name);

        auto canonical = table.table_.find("output.weight");
        if (canonical == table.table_.end()) throw std::runtime_error("output.weight is missing from WeightTable");
        WeightRef& ref = canonical->second;
        if (ref.exl3 != nullptr) throw std::runtime_error("output.weight already has an EXL3 owner");
        if (ref.native_data != nullptr) throw std::runtime_error("output.weight already has a native projection");
        if (descriptor->k != ref.ne0 || descriptor->n != ref.ne1 || descriptor->k <= 0 || descriptor->n <= 0)
            throw std::runtime_error("output.weight shape does not match the EXL3 descriptor");
        if (descriptor->k % 128 != 0 || descriptor->n % 128 != 0)
            throw std::runtime_error("output.weight is not 128-aligned for the EXL3 Hadamard path");

        strata::artifact::exl3::MaterializedMatrix materialized;
        archive.materialize(*descriptor, materialized);
        const auto host_view = materialized.view();
        runtime_ = &Runtime::get();
        max_rows_ = max_rows;
        scratch_elements_ = strata::kernels::exl3::exl3_gemm_scratch_elements(host_view, max_rows_);

        if (!copy_component(trellis_, materialized.trellis, "trellis", err) ||
            !copy_component(suh_, materialized.suh, "suh", err) ||
            !copy_component(svh_, materialized.svh, "svh", err) ||
            !copy_component(bias_, materialized.bias, "bias", err))
            throw std::runtime_error(err);
        if (scratch_elements_ > std::numeric_limits<size_t>::max() / sizeof(float))
            throw std::runtime_error("scratch size overflows size_t");
        view_ = host_view;
        view_.trellis = static_cast<const uint16_t*>(trellis_);
        view_.suh = static_cast<const uint16_t*>(suh_);  // codespell:ignore suh
        view_.svh = static_cast<const uint16_t*>(svh_);
        view_.bias = static_cast<const uint16_t*>(bias_);
        if (!pointers_owned_by(runtime_)) throw std::runtime_error("EXL3 allocation belongs to another runtime");

        ref.exl3 = this;
        attached_ = &ref;
        if (!prepare_stream(nullptr, err)) throw std::runtime_error(err);
        return true;
    } catch (const std::exception& e) {
        if (err.empty()) err = std::string("exl3 dense: ") + e.what();
        if (attached_ != nullptr && attached_->exl3 == this) attached_->exl3 = nullptr;
        attached_ = nullptr;
        reset();
        return false;
    }
}

bool Exl3Dense::prepare_stream(void* stream, std::string& err) {
    err.clear();
    if (runtime_ == nullptr || view_.trellis == nullptr) {
        err = "exl3 dense: the owner is not loaded";
        return false;
    }
    try {
        Runtime* stream_runtime = stream == nullptr ? &Runtime::get() : Runtime::owner_of_stream(stream);
        if (stream_runtime == nullptr || stream_runtime != runtime_) {
            err = "exl3 dense: the stream belongs to another runtime";
            return false;
        }
        sycl::queue& queue = stream_runtime->stream(stream);
        const void* queue_identity = &queue;
        for (const StreamScratch& prepared : streams_)
            if (prepared.queue == queue_identity) return true;
        if (scratch_elements_ > std::numeric_limits<size_t>::max() / sizeof(float)) {
            err = "exl3 dense: scratch size overflows size_t";
            return false;
        }
        void* scratch = nullptr;
        {
            const OnDevice on(stream_runtime->ordinal());
            scratch = strata::gpu::alloc_device(scratch_elements_ * sizeof(float));
        }
        if (scratch == nullptr) {
            err = std::string("exl3 dense: scratch allocation failed: ") + strata::gpu::last_error();
            return false;
        }
        if (Runtime::owner(scratch) != stream_runtime) {
            strata::gpu::free(scratch);
            err = "exl3 dense: scratch allocation belongs to another runtime";
            return false;
        }
        try {
            streams_.push_back(StreamScratch{queue_identity, scratch});
        } catch (...) {
            strata::gpu::free(scratch);
            throw;
        }
        return true;
    } catch (const std::exception& e) {
        err = std::string("exl3 dense: ") + e.what();
        return false;
    }
}

bool Exl3Dense::run(const float* x, int64_t ldx, float* y, int64_t ldy, int64_t rows, void* stream,
                    std::string& err) {
    if (runtime_ == nullptr || view_.trellis == nullptr) {
        err = "exl3 dense: the owner is not loaded";
        return false;
    }
    if (x == nullptr || y == nullptr || rows <= 0 || rows > max_rows_ || ldx < view_.k || ldy < view_.n) {
        err = "exl3 dense: invalid input/output shape";
        return false;
    }
    try {
        Runtime* stream_runtime = stream == nullptr ? &Runtime::get() : Runtime::owner_of_stream(stream);
        if (stream_runtime == nullptr || stream_runtime != runtime_) {
            err = "exl3 dense: the stream belongs to another runtime";
            return false;
        }
        if (Runtime::owner(x) != runtime_ || Runtime::owner(y) != runtime_) {
            err = "exl3 dense: input or output belongs to another runtime";
            return false;
        }
        sycl::queue& queue = stream_runtime->stream(stream);
        if (!pointers_owned_by(stream_runtime)) {
            err = "exl3 dense: the descriptor or scratch belongs to another runtime";
            return false;
        }
        const void* queue_identity = &queue;
        const StreamScratch* prepared = nullptr;
        for (const StreamScratch& candidate : streams_)
            if (candidate.queue == queue_identity) { prepared = &candidate; break; }
        if (prepared == nullptr) {
            err = "exl3 dense: stream was not prepared before graph capture";
            return false;
        }
        const size_t needed = strata::kernels::exl3::exl3_gemm_scratch_elements(view_, rows);
        if (needed > scratch_elements_) {
            err = "exl3 dense: caller scratch is too small for this row count";
            return false;
        }
        strata::kernels::exl3::exl3_gemm_fp32(queue, view_, x, ldx, y, ldy, rows,
                                              static_cast<float*>(prepared->data), scratch_elements_);
        return true;
    } catch (const std::exception& e) {
        err = std::string("exl3 dense: ") + e.what();
        return false;
    }
}

}  // namespace strata::core
