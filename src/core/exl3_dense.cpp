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
#include "strata/kernels/xmx_gemm.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/raw_linear.hpp"

#include <algorithm>
#include <atomic>
#include <cstdlib>
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

/// The bytes every live FP16 weight cache holds, against prefill_fp16_cache_budget_bytes().
std::atomic<uint64_t>& fp16_cache_used() {
    static std::atomic<uint64_t> used{0};
    return used;
}

bool fp16_cache_charge(size_t bytes, uint64_t budget) {
    if (bytes == 0) return true;
    if (static_cast<uint64_t>(bytes) > budget) return false;
    uint64_t used = fp16_cache_used().load(std::memory_order_relaxed);
    for (;;) {
        if (used > budget - static_cast<uint64_t>(bytes)) return false;
        if (fp16_cache_used().compare_exchange_weak(used, used + static_cast<uint64_t>(bytes),
                                                    std::memory_order_relaxed))
            return true;
    }
}

void fp16_cache_release(size_t bytes) {
    if (bytes != 0) fp16_cache_used().fetch_sub(static_cast<uint64_t>(bytes), std::memory_order_relaxed);
}

}  // namespace

uint64_t prefill_fp16_cache_budget_bytes(uint64_t total_bytes) {
    static const int64_t asked_mib = [] {
        const char* text = std::getenv("STRATA_EXL3_PREFILL_FP16_CACHE_MIB");
        if (text == nullptr || text[0] == '\0') return int64_t{-1};
        char* end = nullptr;
        const long long value = std::strtoll(text, &end, 10);
        if (end == text || (end != nullptr && *end != '\0') || value < 0) return int64_t{-1};
        return static_cast<int64_t>(value);
    }();
    if (asked_mib >= 0) return static_cast<uint64_t>(asked_mib) << 20;
    if (total_bytes == 0) return 0;
    return std::clamp<uint64_t>(total_bytes / 8, 256ull << 20, 8ull << 30);
}

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
    if (fp16_w_borrowed_) {
        fp16_w_ = nullptr;  // aliases raw_weights_, freed above
    } else {
        free_device(fp16_w_);
    }
    fp16_cache_release(fp16_w_bytes_);
    fp16_w_bytes_ = 0;
    fp16_w_borrowed_ = false;
    free_device(fp16_x_);
    fp16_x_elems_ = 0;
    {
        void* p = fp16_xh_;
        free_device(p);
        fp16_xh_ = nullptr;
    }
    fp16_xh_elems_ = 0;
    fp16_backoff_ = 0;
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

bool Exl3Dense::input_scratch(void* stream, int64_t rows, float*& plane, std::string& err) {
    err.clear();
    plane = nullptr;
    if (runtime_ == nullptr || raw_ || view_.trellis == nullptr || view_.suh == nullptr) {
        err = "exl3 dense: the owner has no packed input transform";
        return false;
    }
    if (rows <= 0 || rows > max_rows_) {
        err = "exl3 dense: invalid input scratch row count";
        return false;
    }
    try {
        Runtime* stream_runtime = stream == nullptr ? &Runtime::get() : Runtime::owner_of_stream(stream);
        if (stream_runtime == nullptr || stream_runtime != runtime_) {
            err = "exl3 dense: the stream belongs to another runtime";
            return false;
        }
        sycl::queue& queue = stream_runtime->stream(stream);
        if (!pointers_owned_by(stream_runtime)) {
            err = "exl3 dense: the descriptor or scratch belongs to another runtime";
            return false;
        }
        const void* queue_identity = &queue;
        for (const StreamScratch& prepared : streams_)
            if (prepared.queue == queue_identity) {
                if (prepared.data == nullptr) break;
                plane = static_cast<float*>(prepared.data);
                return true;
            }
        err = "exl3 dense: stream was not prepared before graph capture";
        return false;
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

bool Exl3Dense::run_fp16_gemm(const float* x, int64_t ldx, float* y, int64_t ldy, int64_t rows, void* stream,
                              const strata::kernels::exl3::GemvOptions& options, std::string& err) {
    const bool is_raw = raw_;
    if (is_raw ? raw_view_.data == nullptr : view_.trellis == nullptr) {
        err = "exl3 dense fp16: the owner is not loaded";
        return false;
    }
    const int64_t k = is_raw ? raw_view_.k : view_.k;
    const int64_t n = is_raw ? raw_view_.n : view_.n;
    if (x == nullptr || y == nullptr || rows <= 0 || rows > max_rows_ || ldx != k || ldy != n) {
        err = "exl3 dense fp16: tight strides and rows <= max_rows are required";
        return false;
    }
    // A raw owner has no basis stages: `run` ignores `options` for it, and so does this path.
    // Its bias is the FP32 one `linear_fp32` adds after the product.
    const bool use_input = !is_raw && options.input_hadamard && view_.suh != nullptr;
    const bool use_output = !is_raw && options.output_hadamard && view_.svh != nullptr;
    if ((use_input && k % 128 != 0) || (use_output && n % 128 != 0)) {
        err = "exl3 dense fp16: transform widths must be multiples of 128";
        return false;
    }
    if (!is_raw && view_.bias != nullptr && options.add_bias && !use_output) {
        err = "exl3 dense fp16: the bias needs the output transform stage";
        return false;
    }
    if (fp16_backoff_ > 0) {
        --fp16_backoff_;
        err = "exl3 dense fp16: cache allocation failed recently; retry deferred";
        return false;
    }
    // Any cache-build failure frees what it took, charges nothing it did not keep, and backs
    // off for a bounded number of calls instead of latching for the owner's lifetime: room
    // freed later in the process lets a later chunk take the FP16 path after all.
    const auto fail_build = [this](const char* why) {
        if (fp16_w_ != nullptr && !fp16_w_borrowed_) strata::gpu::free(fp16_w_);
        fp16_w_ = nullptr;
        fp16_w_borrowed_ = false;
        fp16_cache_release(fp16_w_bytes_);
        fp16_w_bytes_ = 0;
        fp16_backoff_ = 32;
        (void) why;
        return false;
    };
    try {
        Runtime* stream_runtime = stream == nullptr ? &Runtime::get() : Runtime::owner_of_stream(stream);
        if (stream_runtime == nullptr || stream_runtime != runtime_) {
            err = "exl3 dense fp16: the stream belongs to another runtime";
            return false;
        }
        sycl::queue& queue = stream_runtime->stream(stream);
        if (fp16_w_ == nullptr) {
            const size_t elems = static_cast<size_t>(k) * static_cast<size_t>(n);
            const size_t bytes = elems * sizeof(uint16_t);
            if (is_raw && raw_view_.dtype == strata::kernels::raw::DType::F16) {
                // The raw F16 weights already are W[N, K]: use them in place.  No allocation,
                // no budget charge, so a raw owner takes the FP16 path however full the
                // device is - this is the owner the standard env's raw projections produce.
                fp16_w_ = raw_weights_;
                fp16_w_borrowed_ = true;
            } else {
                const uint64_t total = queue.get_device().get_info<sycl::info::device::global_mem_size>();
                if (!fp16_cache_charge(bytes, prefill_fp16_cache_budget_bytes(total))) {
                    fp16_backoff_ = 32;
                    err = "exl3 dense fp16: the shared cache budget is exhausted; packed products";
                    return false;
                }
                fp16_w_bytes_ = bytes;
                fp16_w_ = strata::gpu::alloc_device(bytes);
                if (fp16_w_ == nullptr) {
                    err = std::string("exl3 dense fp16: cache allocation failed: ") + strata::gpu::last_error();
                    return fail_build("alloc");
                }
                uint16_t* w16 = static_cast<uint16_t*>(fp16_w_);
                if (is_raw) {
                    // Convert the raw weights once, in row chunks of the output width, so the
                    // transient is one chunk and not a second full matrix.
                    const size_t chunk_elems = size_t{1} << 20;
                    const strata::kernels::raw::DType dtype = raw_view_.dtype;
                    const uint16_t* src16 = static_cast<const uint16_t*>(raw_view_.data);
                    const float* src32 = static_cast<const float*>(raw_view_.data);
                    for (size_t off = 0; off < elems; off += chunk_elems) {
                        const size_t count = std::min(chunk_elems, elems - off);
                        const uint16_t* s16 = src16 + off;
                        const float* s32 = src32 + off;
                        uint16_t* dst = w16 + off;
                        queue.parallel_for(sycl::range<1>(count), [=](sycl::id<1> id) {
                            dst[id] = dtype == strata::kernels::raw::DType::BF16
                                          ? strata::kernels::f16_from_f32(strata::kernels::f32_from_bf16(s16[id]))
                                          : strata::kernels::f16_from_f32(s32[id]);
                        });
                    }
                } else if (view_.tile_order == strata::kernels::exl3::TileOrder::canonical) {
                    // Dequantize in K chunks: reconstruct one [chunk_k, n] slice and transpose
                    // it into its column slice of W[N, K].  The transient is the slice, not a
                    // second full [k, n] matrix beside the cache.
                    int64_t chunk_k = std::max<int64_t>(16, (int64_t{1} << 20) / std::max<int64_t>(n, 1));
                    chunk_k = std::min<int64_t>(k, (chunk_k / 16) * 16);
                    void* recon = strata::gpu::alloc_device(static_cast<size_t>(chunk_k) * static_cast<size_t>(n) *
                                                            sizeof(uint16_t));
                    void* trans = strata::gpu::alloc_device(static_cast<size_t>(chunk_k) * static_cast<size_t>(n) *
                                                            sizeof(uint16_t));
                    if (recon == nullptr || trans == nullptr) {
                        strata::gpu::free(recon);
                        strata::gpu::free(trans);
                        err = std::string("exl3 dense fp16: chunk allocation failed: ") + strata::gpu::last_error();
                        return fail_build("chunk");
                    }
                    const int64_t tiles_n = n / 16;
                    const size_t tile_elems = static_cast<size_t>(16 * view_.bits);
                    for (int64_t k_off = 0; k_off < k; k_off += chunk_k) {
                        const int64_t rows_k = std::min(chunk_k, k - k_off);
                        strata::kernels::exl3::MatrixView slice = view_;
                        slice.k = rows_k;
                        slice.trellis = view_.trellis +
                                        static_cast<size_t>(k_off / 16) * static_cast<size_t>(tiles_n) * tile_elems;
                        slice.suh = nullptr;  // codespell:ignore suh
                        slice.svh = nullptr;
                        slice.bias = nullptr;
                        const sycl::event built = strata::kernels::exl3::exl3_transpose_f16(
                            queue, static_cast<const uint16_t*>(recon), static_cast<uint16_t*>(trans), rows_k, n,
                            {strata::kernels::exl3::exl3_reconstruct(queue, slice, static_cast<uint16_t*>(recon))});
                        (void) built;
                        uint16_t* dst = w16 + static_cast<size_t>(k_off);
                        const uint16_t* src = static_cast<const uint16_t*>(trans);
                        const int64_t stride_k = k;
                        queue.submit([&](sycl::handler& h) {
                            h.depends_on(built);
                            h.parallel_for(sycl::range<1>(static_cast<size_t>(rows_k) * static_cast<size_t>(n)),
                                           [=](sycl::id<1> id) {
                                               const size_t flat = static_cast<size_t>(id[0]);
                                               const size_t col = flat / static_cast<size_t>(rows_k);
                                               const size_t row = flat % static_cast<size_t>(rows_k);
                                               dst[col * static_cast<size_t>(stride_k) + row] =
                                                   src[col * static_cast<size_t>(rows_k) + row];
                                           });
                        });
                    }
                    queue.wait();
                    strata::gpu::free(recon);
                    strata::gpu::free(trans);
                } else {
                    // A CPU tile order the reconstruct kernel cannot read slice-wise: keep the
                    // whole-matrix build (transient [k, n] beside the cache), still under the
                    // same budget and failure rules.
                    void* recon = strata::gpu::alloc_device(bytes);
                    if (recon == nullptr) {
                        err = std::string("exl3 dense fp16: dequant allocation failed: ") + strata::gpu::last_error();
                        return fail_build("recon");
                    }
                    const sycl::event built = strata::kernels::exl3::exl3_transpose_f16(
                        queue, static_cast<const uint16_t*>(recon), w16, k, n,
                        {strata::kernels::exl3::exl3_reconstruct(queue, view_, static_cast<uint16_t*>(recon))});
                    (void) built;
                    queue.wait();
                    strata::gpu::free(recon);
                }
            }
        }
        // Activation/product scratch sized for this call's rows, grown only when a later call
        // is wider - not max_rows_ up front for every owner.
        const size_t need_x = static_cast<size_t>(rows) * static_cast<size_t>(k);
        const size_t need_xh = static_cast<size_t>(rows) * static_cast<size_t>(k + n);
        if (fp16_x_elems_ < need_x) {
            void* next = strata::gpu::alloc_device(need_x * sizeof(uint16_t));
            if (next == nullptr) {
                fp16_backoff_ = 32;
                err = std::string("exl3 dense fp16: activation allocation failed: ") + strata::gpu::last_error();
                return false;
            }
            free_device(fp16_x_);
            fp16_x_ = next;
            fp16_x_elems_ = need_x;
        }
        if (fp16_xh_elems_ < need_xh) {
            void* next = strata::gpu::alloc_device(need_xh * sizeof(float));
            if (next == nullptr) {
                fp16_backoff_ = 32;
                err = std::string("exl3 dense fp16: product allocation failed: ") + strata::gpu::last_error();
                return false;
            }
            void* old = fp16_xh_;
            fp16_xh_ = static_cast<float*>(next);
            if (old != nullptr) strata::gpu::free(old);
            fp16_xh_elems_ = need_xh;
        }
        float* z = fp16_xh_ + static_cast<size_t>(rows) * static_cast<size_t>(k);
        const float* src = x;
        if (use_input) {
            strata::kernels::exl3::exl3_input_hadamard_rows(queue, view_, x, ldx, fp16_xh_, rows);
            src = fp16_xh_;
        }
        // The FP16 product itself: the same own-kernels product the prefill GEMM path falls
        // back to (strata/prefill/gemm.cpp's Gemm::f16 without oneMath).  Gemm itself lives in
        // strata_prefill, which links this library, so the engine calls the kernels directly.
        {
            const float* x32 = src;
            uint16_t* x16 = static_cast<uint16_t*>(fp16_x_);
            const size_t count = static_cast<size_t>(rows) * static_cast<size_t>(k);
            queue.parallel_for(sycl::range<1>(count), [=](sycl::id<1> id) {
                x16[id] = strata::kernels::f16_from_f32(x32[id]);
            });
        }
        const uint16_t* x16 = static_cast<const uint16_t*>(fp16_x_);
        const uint16_t* w16 = static_cast<const uint16_t*>(fp16_w_);
        float* out = use_output ? z : y;
        if (is_raw && raw_view_.bias == nullptr) out = y;
        if (strata::kernels::xmx_gemm_ok(x16, w16, out, n, k, n))
            strata::kernels::xmx_gemm(strata::kernels::XmxType::f16, x16, w16, out, rows, n, k, n, stream);
        else
            strata::kernels::gemm_rows(strata::kernels::XmxType::f16, x16, w16, out, rows, n, k, n, stream);
        if (use_output) {
            strata::kernels::exl3::exl3_output_hadamard_rows(queue, view_, z, y, ldy, rows, options.add_bias);
        } else if (is_raw && raw_view_.bias != nullptr) {
            // Add the raw FP32 bias the packed output stage would have added.
            float* dst = y;
            const float* bias = raw_view_.bias;
            if (out != y && !strata::gpu::copy_async(y, out, static_cast<size_t>(rows) * static_cast<size_t>(n) * sizeof(float), stream)) {
                err = "exl3 dense fp16: output copy failed";
                return false;
            }
            queue.parallel_for(sycl::range<1>(static_cast<size_t>(rows) * static_cast<size_t>(n)),
                               [=](sycl::id<1> id) {
                                   dst[id] += bias[id % static_cast<size_t>(n)];
                               });
        } else if (out != y && !strata::gpu::copy_async(y, out, static_cast<size_t>(rows) * static_cast<size_t>(n) * sizeof(float), stream)) {
            err = "exl3 dense fp16: output copy failed";
            return false;
        }
        return true;
    } catch (const std::exception& e) {
        err = std::string("exl3 dense fp16: ") + e.what();
        return false;
    }
}
}  // namespace strata::core
