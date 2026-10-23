// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "strata/core/exl3_dense.hpp"

#include "strata/artifact/exl3.hpp"
#include "strata/artifact/exl3_raw.hpp"
#include "strata/artifact/exl3_remap.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/runtime.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/exl3.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/raw_linear.hpp"

#include <limits>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {
namespace {
void free_device(void*& pointer) {
    if (pointer != nullptr) {
        strata::gpu::free(pointer);
        pointer = nullptr;
    }
}

bool copy_bytes(void*& destination, const std::vector<uint8_t>& source, const char* label, std::string& err) {
    if (source.empty()) {
        err = std::string("exl3 dense: ") + label + " is empty";
        return false;
    }
    destination = strata::gpu::alloc_device(source.size());
    if (destination == nullptr || !strata::gpu::copy(destination, source.data(), source.size())) {
        if (destination != nullptr) strata::gpu::free(destination);
        destination = nullptr;
        err = std::string("exl3 dense: uploading ") + label + " failed: " + strata::gpu::last_error();
        return false;
    }
    return true;
}

uint16_t read_u16(const uint8_t* data) {
    return static_cast<uint16_t>(data[0]) | static_cast<uint16_t>(data[1] << 8);
}

uint32_t read_u32(const uint8_t* data) {
    return static_cast<uint32_t>(data[0]) | (static_cast<uint32_t>(data[1]) << 8) |
           (static_cast<uint32_t>(data[2]) << 16) | (static_cast<uint32_t>(data[3]) << 24);
}

bool raw_dtype_of(const std::string& dtype, strata::kernels::raw::DType& result) {
    if (dtype == "F16") result = strata::kernels::raw::DType::F16;
    else if (dtype == "BF16") result = strata::kernels::raw::DType::BF16;
    else if (dtype == "F32") result = strata::kernels::raw::DType::F32;
    else return false;
    return true;
}

bool materialize_raw_bias(const strata::artifact::exl3::TensorDescriptor& tensor, const std::vector<uint8_t>& raw,
                          std::vector<float>& result,
                          std::string& err) {
    if (tensor.shape.size() != 1 || tensor.shape[0] > std::numeric_limits<size_t>::max()) {
        err = "exl3 dense: raw bias must be one-dimensional";
        return false;
    }
    const size_t count = static_cast<size_t>(tensor.shape[0]);
    result.resize(count);
    if (tensor.dtype == "F16") {
        if (raw.size() != count * 2) { err = "exl3 dense: raw F16 bias size"; return false; }
        for (size_t i = 0; i < count; ++i) result[i] = strata::kernels::f32_from_f16(read_u16(raw.data() + i * 2));
    } else if (tensor.dtype == "BF16") {
        if (raw.size() != count * 2) { err = "exl3 dense: raw BF16 bias size"; return false; }
        for (size_t i = 0; i < count; ++i) result[i] = strata::kernels::f32_from_bf16(read_u16(raw.data() + i * 2));
    } else if (tensor.dtype == "F32") {
        if (raw.size() != count * 4) { err = "exl3 dense: raw F32 bias size"; return false; }
        for (size_t i = 0; i < count; ++i) {
            const uint32_t bits = read_u32(raw.data() + i * 4);
            std::memcpy(&result[i], &bits, sizeof bits);
        }
    } else {
        err = "exl3 dense: raw bias dtype is not F16/BF16/F32";
        return false;
    }
    return true;
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

/// Whether the uploaded F16 weights hold normal halves and signed zeros only.  The answer tells
/// the kernel that the device's half to float conversion is exact for every one of them.
bool plain_f16_weights(void* weights, size_t bytes) {
    if (bytes == 0) return true;
    return strata::kernels::raw::plain_f16_halves(strata::core::Runtime::get().compute(), weights, bytes);
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
    free_device(raw_weights_);
    free_device(raw_bias_);
    for (StreamScratch& stream : streams_) free_device(stream.data);
    streams_.clear();
    view_ = {};
    raw_view_ = {};
    scratch_elements_ = 0;
    max_rows_ = 0;
    runtime_ = nullptr;
    raw_ = false;
    attached_ = nullptr;
    detached_table_.reset();
}

bool Exl3Dense::pointers_owned_by(const Runtime* runtime) const {
    if (runtime == nullptr) return false;
    const void* pointers[] = {view_.trellis, view_.suh, view_.svh, view_.bias, raw_view_.data, raw_view_.bias};  // codespell:ignore suh
    for (const void* pointer : pointers)
        if (pointer != nullptr && Runtime::owner(pointer) != runtime) return false;
    for (const StreamScratch& stream : streams_)
        if (stream.data != nullptr && Runtime::owner(stream.data) != runtime) return false;
    return true;
}

bool Exl3Dense::load(const std::filesystem::path& directory, const std::string& hf_name, WeightTable& table,
                     std::string& err, int64_t max_rows) {
    return load(directory, hf_name, "output.weight", table, err, max_rows);
}

bool Exl3Dense::load(const std::filesystem::path& directory, const std::string& hf_name,
                     const std::string& canonical_name, WeightTable& table, std::string& err, int64_t max_rows) {
    if (runtime_ != nullptr || attached_ != nullptr) {
        err = "exl3 dense: the owner is already loaded";
        return false;
    }
    try {
        const strata::artifact::exl3::Archive archive(directory);
        return load(archive, hf_name, canonical_name, table, err, max_rows);
    } catch (const std::exception& error) {
        err = std::string("exl3 dense: ") + error.what();
        return false;
    }
}

bool Exl3Dense::load(const strata::artifact::exl3::Archive& archive, const std::string& hf_name, WeightTable& table,
                     std::string& err, int64_t max_rows) {
    return load(archive, hf_name, "output.weight", table, err, max_rows);
}

bool Exl3Dense::load(const strata::artifact::exl3::Archive& archive, const std::string& hf_name,
                     const std::string& canonical_name, WeightTable& table, std::string& err, int64_t max_rows) {
    return load_remapped(archive, hf_name, canonical_name, {}, {}, table, err, max_rows);
}

bool Exl3Dense::load_remapped(const strata::artifact::exl3::Archive& archive, const std::string& hf_name,
                              const std::string& canonical_name, const std::vector<int64_t>& input_blocks,
                              const std::vector<int64_t>& output_blocks, WeightTable& table, std::string& err,
                              int64_t max_rows) {
    err.clear();
    if (runtime_ != nullptr || attached_ != nullptr) {
        err = "exl3 dense: the owner is already loaded";
        return false;
    }
    if (hf_name.empty() || canonical_name.empty() || max_rows <= 0) {
        err = "exl3 dense: HF name, canonical name and max_rows are required";
        return false;
    }
    try {
        auto canonical = table.table_.find(canonical_name);
        if (canonical == table.table_.end())
            throw std::runtime_error(canonical_name + " is missing from WeightTable");
        WeightRef& ref = canonical->second;
        if (ref.exl3 != nullptr) throw std::runtime_error(canonical_name + " already has an EXL3 owner");
        if (ref.native_data != nullptr) throw std::runtime_error(canonical_name + " already has a native projection");
        const strata::artifact::exl3::LinearDescriptor* descriptor = archive.find_linear(hf_name);
        if (descriptor == nullptr) {
            if (!input_blocks.empty() || !output_blocks.empty())
                throw std::runtime_error("raw projections require explicit raw-axis mapping");
            const strata::artifact::exl3::TensorDescriptor* weight = archive.find_tensor(hf_name + ".weight");
            if (weight == nullptr) {
                throw std::runtime_error("head weight is missing from both EXL3 and raw archive forms: " + hf_name);
            }
            if (ref.ne0 <= 0 || ref.ne1 <= 0 || weight->shape.size() != 2 ||
                weight->shape[0] != static_cast<uint64_t>(ref.ne1) ||
                weight->shape[1] != static_cast<uint64_t>(ref.ne0))
                throw std::runtime_error("raw " + canonical_name + " shape does not match the HF tensor");
            strata::kernels::raw::DType dtype;
            if (!raw_dtype_of(weight->dtype, dtype))
                throw std::runtime_error("raw " + canonical_name + " dtype is not F16/BF16/F32");
            std::vector<uint8_t> raw;
            archive.read_tensor(*weight, raw);
            if (!copy_bytes(raw_weights_, raw, "raw weight", err)) throw std::runtime_error(err);
            const strata::artifact::exl3::TensorDescriptor* bias = archive.find_tensor(hf_name + ".bias");
            std::vector<float> bias_values;
            if (bias != nullptr) {
                if (bias->shape != std::vector<uint64_t>{static_cast<uint64_t>(ref.ne1)})
                    throw std::runtime_error("raw " + canonical_name + " bias shape does not match output width");
                std::vector<uint8_t> bias_raw;
                archive.read_tensor(*bias, bias_raw);
                if (!materialize_raw_bias(*bias, bias_raw, bias_values, err)) throw std::runtime_error(err);
                std::vector<uint8_t> bias_bytes(bias_values.size() * sizeof(float));
                std::memcpy(bias_bytes.data(), bias_values.data(), bias_bytes.size());
                if (!bias_values.empty() && !copy_bytes(raw_bias_, bias_bytes, "raw bias", err))
                    throw std::runtime_error(err);
            }
            runtime_ = &Runtime::get();
            raw_view_ = {raw_weights_, raw_bias_ == nullptr ? nullptr : static_cast<const float*>(raw_bias_), ref.ne0,
                         ref.ne1, dtype};
            if (dtype == strata::kernels::raw::DType::F16)
                raw_view_.plain_halves = plain_f16_weights(raw_weights_, raw.size());
            raw_ = true;
            max_rows_ = max_rows;
            ref.exl3 = this;
            attached_ = &ref;
            if (!prepare_stream(nullptr, err)) throw std::runtime_error(err);
            return true;
        }
        if (descriptor->k != ref.ne0 || (output_blocks.empty() && descriptor->n != ref.ne1) ||
            descriptor->k <= 0 || descriptor->n <= 0)
            throw std::runtime_error(canonical_name + " shape does not match the EXL3 descriptor");
        if (descriptor->k % 128 != 0 || descriptor->n % 128 != 0)
            throw std::runtime_error(canonical_name + " is not 128-aligned for the EXL3 Hadamard path");

        strata::artifact::exl3::MaterializedMatrix materialized;
        archive.materialize(*descriptor, materialized);
        if (!input_blocks.empty() || !output_blocks.empty()) {
            strata::artifact::exl3::MaterializedMatrix remapped;
            if (!strata::artifact::exl3::remap_matrix_blocks(materialized, input_blocks, output_blocks, remapped, err))
                throw std::runtime_error(err);
            materialized = std::move(remapped);
        }
        if (materialized.k != ref.ne0 || materialized.n != ref.ne1)
            throw std::runtime_error(hf_name + " remapped shape [" + std::to_string(materialized.k) + "," +
                                     std::to_string(materialized.n) + "] does not match target [" +
                                     std::to_string(ref.ne0) + "," + std::to_string(ref.ne1) + "]");
        const auto host_view = materialized.view();
        runtime_ = &Runtime::get();
        max_rows_ = max_rows;
        scratch_elements_ = strata::kernels::exl3::exl3_gemm_scratch_elements(host_view, max_rows_);

        if (!copy_component(trellis_, materialized.trellis, "trellis", err) ||
            !copy_component(suh_, materialized.suh, "suh", err) ||  // codespell:ignore suh
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

bool Exl3Dense::load_detached(const strata::artifact::exl3::Archive& archive, const std::string& hf_name,
                              int64_t input_width, int64_t output_width, const std::vector<int64_t>& input_blocks,
                              const std::vector<int64_t>& output_blocks, std::string& err, int64_t max_rows) {
    if (runtime_ != nullptr || attached_ != nullptr) {
        err = "exl3 dense: the owner is already loaded";
        return false;
    }
    if (input_width <= 0 || output_width <= 0 ||
        input_width > std::numeric_limits<int64_t>::max() / output_width) {
        err = "exl3 dense: invalid detached projection dimensions";
        return false;
    }
    auto table = std::make_unique<WeightTable>();
    WeightRef ref;
    ref.ne0 = input_width;
    ref.ne1 = output_width;
    ref.elements = input_width * output_width;
    table->table_.emplace("exl3.detached", ref);
    if (!load_remapped(archive, hf_name, "exl3.detached", input_blocks, output_blocks, *table, err, max_rows))
        return false;
    detached_table_ = std::move(table);
    return true;
}

bool Exl3Dense::load_raw_detached(const strata::artifact::exl3::RawTensor& weights,
                                  const strata::artifact::exl3::RawTensor* bias, std::string& err,
                                  int64_t max_rows) {
    err.clear();
    if (runtime_ != nullptr || attached_ != nullptr) {
        err = "exl3 dense: the owner is already loaded";
        return false;
    }
    try {
        strata::kernels::raw::DType dtype;
        if (!raw_dtype_of(weights.dtype, dtype) || weights.shape.size() != 2 || max_rows <= 0)
            throw std::invalid_argument("raw projection requires a F16/BF16/F32 matrix and positive row capacity");
        const uint64_t n = weights.shape[0], k = weights.shape[1];
        const uint64_t width = weights.dtype == "F32" ? 4 : 2;
        const uint64_t limit = static_cast<uint64_t>(std::numeric_limits<ptrdiff_t>::max()) / width;
        if (n == 0 || k == 0 || n > limit / k || weights.data.size() != n * k * width)
            throw std::invalid_argument("raw projection payload does not match its dimensions");
        std::vector<float> bias_values;
        if (bias != nullptr) {
            if (bias->shape != std::vector<uint64_t>{n})
                throw std::invalid_argument("raw projection bias does not match output width");
            if (!strata::artifact::exl3::raw_to_f32(*bias, bias_values, err)) throw std::runtime_error(err);
        }
        if (!copy_bytes(raw_weights_, weights.data, "raw weight", err)) throw std::runtime_error(err);
        if (!bias_values.empty()) {
            std::vector<uint8_t> bytes(bias_values.size() * sizeof(float));
            std::memcpy(bytes.data(), bias_values.data(), bytes.size());
            if (!copy_bytes(raw_bias_, bytes, "raw bias", err)) throw std::runtime_error(err);
        }
        runtime_ = &Runtime::get();
        raw_view_ = {raw_weights_, static_cast<const float*>(raw_bias_), static_cast<int64_t>(k),
                     static_cast<int64_t>(n), dtype};
        if (dtype == strata::kernels::raw::DType::F16)
            raw_view_.plain_halves = plain_f16_weights(raw_weights_, weights.data.size());
        raw_ = true;
        max_rows_ = max_rows;
        if (!pointers_owned_by(runtime_)) throw std::runtime_error("raw allocations belong to another runtime");
        if (!prepare_stream(nullptr, err)) throw std::runtime_error(err);
        return true;
    } catch (const std::exception& e) {
        if (err.empty()) err = std::string("exl3 dense: ") + e.what();
        reset();
        return false;
    }
}

bool Exl3Dense::prepare_stream(void* stream, std::string& err) {
    err.clear();
    if (runtime_ == nullptr || (raw_ ? raw_view_.data == nullptr : view_.trellis == nullptr)) {
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
        if (raw_) {
            streams_.push_back(StreamScratch{queue_identity, nullptr});
            return true;
        }
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
    return run(x, ldx, y, ldy, rows, stream, strata::kernels::exl3::GemvOptions{}, err);
}

bool Exl3Dense::run(const float* x, int64_t ldx, float* y, int64_t ldy, int64_t rows, void* stream,
                    const strata::kernels::exl3::GemvOptions& options, std::string& err) {
    if (runtime_ == nullptr || (raw_ ? raw_view_.data == nullptr : view_.trellis == nullptr)) {
        err = "exl3 dense: the owner is not loaded";
        return false;
    }
    const int64_t input_width = raw_ ? raw_view_.k : view_.k;
    const int64_t output_width = raw_ ? raw_view_.n : view_.n;
    if (x == nullptr || y == nullptr || rows <= 0 || rows > max_rows_ || input_width <= 0 || output_width <= 0 ||
        ldx < input_width || ldy < output_width) {
        err = "exl3 dense: invalid input/output shape";
        return false;
    }
    try {
        Runtime* stream_runtime = stream == nullptr ? &Runtime::get() : Runtime::owner_of_stream(stream);
        if (stream_runtime == nullptr || stream_runtime != runtime_) {
            err = "exl3 dense: the stream belongs to another runtime";
            return false;
        }
        // The input and the output must be usable by this runtime's queue.  A device allocation of
        // another device is not; host USM is, because the device reads it over the interconnect.  A
        // multi-stage model hands a stage the host buffers another stage allocated, so the owner's
        // runtime alone does not decide it.
        const auto usable_here = [this](const void* pointer) {
            if (pointer == nullptr) return false;
            if (Runtime::owner(pointer) == runtime_) return true;
            return sycl::get_pointer_type(pointer, runtime_->context()) == sycl::usm::alloc::host;
        };
        if (!usable_here(x) || !usable_here(y)) {
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
        if (raw_) {
            strata::kernels::raw::linear_fp32(queue, raw_view_, x, ldx, y, ldy, rows);
            return true;
        }
        const size_t needed = strata::kernels::exl3::exl3_gemm_scratch_elements(view_, rows, options);
        if (needed > scratch_elements_) {
            err = "exl3 dense: caller scratch is too small for this row count";
            return false;
        }
        strata::kernels::exl3::exl3_gemm_fp32(queue, view_, x, ldx, y, ldy, rows,
                                              static_cast<float*>(prepared->data), scratch_elements_, options);
        return true;
    } catch (const std::exception& e) {
        err = std::string("exl3 dense: ") + e.what();
        return false;
    }
}

}  // namespace strata::core
