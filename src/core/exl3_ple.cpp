// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "strata/core/exl3_ple.hpp"

#include "strata/artifact/exl3_raw.hpp"
#include "strata/core/exl3_dense.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/ngram.hpp"

#include <nlohmann/json.hpp>

#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace strata::core {
namespace {

using Json = nlohmann::json;
using Archive = strata::artifact::exl3::Archive;
using RawTensor = strata::artifact::exl3::RawTensor;
using TensorDescriptor = strata::artifact::exl3::TensorDescriptor;
using ModelMap = strata::artifact::exl3::ModelMap;

void free_device(void *&pointer) noexcept {
    if (pointer != nullptr) {
        strata::gpu::free(pointer);
        pointer = nullptr;
    }
}

bool is_positive_integer(const Json &object, const char *key, int64_t &value) {
    const auto it = object.find(key);
    if (it == object.end() || (!it->is_number_integer() && !it->is_number_unsigned()))
        return false;
    try {
        value = it->get<int64_t>();
    } catch (...) {
        return false;
    }
    return value > 0;
}

bool exact_shape(const std::vector<uint64_t> &actual, std::initializer_list<uint64_t> expected) {
    return actual == std::vector<uint64_t>(expected);
}

std::string source_name(const std::string &source_prefix, int source_layer, const char *role) {
    return source_prefix + "." + std::to_string(source_layer) + ".ple." + role;
}

bool dtype_is_raw(const std::string &dtype) { return dtype == "F16" || dtype == "BF16" || dtype == "F32"; }

} // namespace

Exl3Ple::Exl3Ple() = default;
Exl3Ple::~Exl3Ple() { close(); }

bool Exl3Ple::validate_config(const Archive &archive, int &source_layer, std::string &err) const {
    try {
        const Json config = Json::parse(archive.config_json(), nullptr, false);
        const Json *text = &config;
        const auto text_it = config.find("text_config");
        if (text_it != config.end()) {
            if (!text_it->is_object()) {
                err = "EXL3 PLE: text_config must be an object";
                return false;
            }
            text = &*text_it;
        }

        int64_t embed_dim = 0;
        int64_t conv_kernel = 0;
        int64_t ngram_size = 0;
        int64_t heads_per_ngram = 0;
        if (!is_positive_integer(*text, "ple_embed_dim", embed_dim) || embed_dim != kernels::NG_N_EMBD) {
            err = "EXL3 PLE: ple_embed_dim must equal " + std::to_string(kernels::NG_N_EMBD);
            return false;
        }
        if (!is_positive_integer(*text, "ple_conv_kernel_size", conv_kernel) ||
            conv_kernel != kernels::PLE_CONV_KERNEL) {
            err = "EXL3 PLE: ple_conv_kernel_size must equal " + std::to_string(kernels::PLE_CONV_KERNEL);
            return false;
        }
        if (!is_positive_integer(*text, "ngram_size", ngram_size) || ngram_size != kernels::NGRAM_SIZE) {
            err = "EXL3 PLE: ngram_size must equal " + std::to_string(kernels::NGRAM_SIZE);
            return false;
        }
        if (!is_positive_integer(*text, "heads_per_ngram", heads_per_ngram) ||
            heads_per_ngram != kernels::HEADS_PER_NGRAM) {
            err = "EXL3 PLE: heads_per_ngram must equal " + std::to_string(kernels::HEADS_PER_NGRAM);
            return false;
        }
        const auto layers_it = text->find("ple_layer_ids");
        if (layers_it == text->end() || !layers_it->is_array() || layers_it->size() != 1 ||
            (!(*layers_it)[0].is_number_integer() && !(*layers_it)[0].is_number_unsigned())) {
            err = "EXL3 PLE: ple_layer_ids must contain exactly the 1-based layer id 2";
            return false;
        }
        const int64_t layer_id = (*layers_it)[0].get<int64_t>();
        if (layer_id != 2) {
            err = "EXL3 PLE: only ple_layer_ids=[2] is supported";
            return false;
        }
        source_layer = static_cast<int>(layer_id - 1);
        return true;
    } catch (const std::exception &e) {
        err = std::string("EXL3 PLE: invalid config.json: ") + e.what();
        return false;
    }
}

bool Exl3Ple::upload_bytes(const std::vector<uint8_t> &data, void *&destination, const char *label, std::string &err) {
    if (data.empty()) {
        err = std::string("EXL3 PLE: ") + label + " is empty";
        return false;
    }
    destination = strata::gpu::alloc_device(data.size());
    if (destination == nullptr || !strata::gpu::copy(destination, data.data(), data.size())) {
        free_device(destination);
        err = std::string("EXL3 PLE: uploading ") + label + " failed: " + strata::gpu::last_error();
        return false;
    }
    return true;
}

bool Exl3Ple::upload_f32(const float *data, size_t count, void *&destination, const char *label, std::string &err) {
    if (data == nullptr || count == 0 || count > std::numeric_limits<size_t>::max() / sizeof(float)) {
        err = std::string("EXL3 PLE: invalid ") + label + " buffer";
        return false;
    }
    destination = strata::gpu::alloc_device(count * sizeof(float));
    if (destination == nullptr || !strata::gpu::copy(destination, data, count * sizeof(float))) {
        free_device(destination);
        err = std::string("EXL3 PLE: uploading ") + label + " failed: " + strata::gpu::last_error();
        return false;
    }
    return true;
}

bool Exl3Ple::load_norm(const Archive &archive, const std::string &name, void *&destination, const char *label,
                        std::string &err) {
    const TensorDescriptor *descriptor = archive.find_tensor(name + ".weight");
    if (descriptor == nullptr) {
        err = std::string("EXL3 PLE: missing ") + label + " tensor " + name + ".weight";
        return false;
    }
    if (!exact_shape(descriptor->shape, {static_cast<uint64_t>(kernels::NG_HC_DIM)})) {
        err = std::string("EXL3 PLE: ") + label + " shape must be [" + std::to_string(kernels::NG_HC_DIM) + "]";
        return false;
    }
    if (!dtype_is_raw(descriptor->dtype)) {
        err = std::string("EXL3 PLE: ") + label + " dtype must be F16, BF16, or F32";
        return false;
    }
    RawTensor materialized;
    if (!strata::artifact::exl3::materialize_raw(archive, *descriptor, {},
                                                 strata::artifact::exl3::ValueTransform::add_one, materialized, err)) {
        err = std::string("EXL3 PLE: ") + label + ": " + err;
        return false;
    }
    std::vector<float> values;
    if (!strata::artifact::exl3::raw_to_f32(materialized, values, err)) {
        err = std::string("EXL3 PLE: ") + label + ": " + err;
        return false;
    }
    return upload_f32(values.data(), values.size(), destination, label, err);
}

bool Exl3Ple::load_projection(const Archive &archive, const std::string &name, int64_t input_width,
                              int64_t output_width, int64_t max_rows, std::unique_ptr<Exl3Dense> &destination,
                              std::string &err) {
    // Exl3Dense::load_detached handles both archive representations.  The raw
    // branch copies its original F16/BF16/F32 bytes; the packed branch keeps
    // the validated trellis and codebook path.  Keeping that discriminator in
    // one owner avoids a silent raw-to-BF16 conversion here.
    auto dense = std::make_unique<Exl3Dense>();
    if (!dense->load_detached(archive, name, input_width, output_width, {}, {}, err, max_rows)) {
        err = "EXL3 PLE: " + name + ": " + err;
        return false;
    }
    destination = std::move(dense);
    return true;
}

bool Exl3Ple::load(const Archive &archive, int device, int64_t max_rows, std::string &err) {
    ModelMap map;
    if (!strata::artifact::exl3::build_model_map(archive, map, err))
        return false;
    return load(archive, map, device, max_rows, err);
}

bool Exl3Ple::load(const Archive &archive, const ModelMap &map, int device, int64_t max_rows, std::string &err) {
    return load_with_map(archive, map, device, max_rows, err);
}

bool Exl3Ple::load_with_map(const Archive &archive, const ModelMap &map, int device, int64_t max_rows,
                            std::string &err) {
    err.clear();
    if (device < 0 || max_rows <= 0) {
        err = "EXL3 PLE: device and positive max_rows are required";
        return false;
    }
    if (map.source_prefix.empty() || map.source_prefix.find(".layers") == std::string::npos) {
        err = "EXL3 PLE: ModelMap has no validated layer source prefix";
        return false;
    }
    int source_layer = -1;
    if (!validate_config(archive, source_layer, err))
        return false;
    if (map.geometry.layers > 0 && source_layer >= map.geometry.layers) {
        err = "EXL3 PLE: configured layer id is outside ModelMap geometry";
        return false;
    }

    const std::string prefix = map.source_prefix + "." + std::to_string(source_layer) + ".ple.";
    for (const char* role : {"key_proj", "value_proj"}) {
        const std::string name = prefix + role;
        if (archive.find_linear(name) == nullptr && archive.find_tensor(name + ".weight") == nullptr) {
            err = "EXL3 PLE: missing projection " + name;
            return false;
        }
    }
    if (runtime_ != nullptr || key_ != nullptr || value_ != nullptr) {
        err = "EXL3 PLE: owner is already loaded";
        return false;
    }
    try {
        const OnDevice on(device);
        runtime_ = &Runtime::get();
        device_ = device;
        if (!load_projection(archive, source_name(map.source_prefix, source_layer, "key_proj"), kernels::NG_N_EMBD,
                             kernels::NG_HC_DIM, max_rows, key_, err) ||
            !load_projection(archive, source_name(map.source_prefix, source_layer, "value_proj"), kernels::NG_N_EMBD,
                             kernels::NG_N_EMBD, max_rows, value_, err))
            throw std::runtime_error(err);
        const size_t key_bytes = static_cast<size_t>(kernels::NG_HC_DIM) * sizeof(float);
        const size_t value_bytes = static_cast<size_t>(kernels::NG_N_EMBD) * sizeof(float);
        pre_key_ = strata::gpu::alloc_device(key_bytes);
        pre_value_ = strata::gpu::alloc_device(value_bytes);
        if (pre_key_ == nullptr || pre_value_ == nullptr || !strata::gpu::memset(pre_key_, 0, key_bytes) ||
            !strata::gpu::memset(pre_value_, 0, value_bytes))
            throw std::runtime_error(std::string("EXL3 PLE: preprojection buffer allocation failed: ") +
                                     strata::gpu::last_error());

        if (!load_norm(archive, prefix + "norm_key", norm_key_, "norm_key", err) ||
            !load_norm(archive, prefix + "norm_query", norm_query_, "norm_query", err) ||
            !load_norm(archive, prefix + "norm_conv", norm_conv_, "norm_conv", err))
            throw std::runtime_error(err);

        const std::string conv_name = prefix + "conv1d";
        const TensorDescriptor *conv = archive.find_tensor(conv_name + ".weight");
        if (conv == nullptr)
            throw std::runtime_error("EXL3 PLE: missing conv1d.weight tensor " + conv_name);
        if (conv->dtype != "F16")
            throw std::runtime_error("EXL3 PLE: conv1d.weight must remain F16; "
                                     "implicit narrowing is unsupported");
        if (!exact_shape(conv->shape, {static_cast<uint64_t>(kernels::NG_HC_DIM), 1,
                                       static_cast<uint64_t>(kernels::PLE_CONV_KERNEL)}))
            throw std::runtime_error("EXL3 PLE: conv1d.weight must have shape [10240,1,4]");
        RawTensor conv_raw;
        if (!strata::artifact::exl3::materialize_raw(archive, *conv, {},
                                                     strata::artifact::exl3::ValueTransform::identity, conv_raw, err))
            throw std::runtime_error("EXL3 PLE: conv1d: " + err);
        if (!upload_bytes(conv_raw.data, conv1d_f16_, "conv1d", err))
            throw std::runtime_error(err);

        weights_ = {};
        weights_.norm_key = static_cast<const float *>(norm_key_);
        weights_.norm_query = static_cast<const float *>(norm_query_);
        weights_.norm_conv = static_cast<const float *>(norm_conv_);
        weights_.conv1d_f16 = static_cast<const uint16_t *>(conv1d_f16_);
        weights_.pre_key = static_cast<const float *>(pre_key_);
        weights_.pre_value = static_cast<const float *>(pre_value_);
        if (!prepare_stream(nullptr, err))
            throw std::runtime_error(err);
        return true;
    } catch (const std::exception &e) {
        if (err.empty())
            err = std::string("EXL3 PLE: ") + e.what();
        close();
        return false;
    }
}

bool Exl3Ple::prepare_stream(void *stream, std::string &err) {
    err.clear();
    if (!loaded()) {
        err = "EXL3 PLE: owner is not loaded";
        return false;
    }
    const OnDevice on(device_);
    Runtime *stream_runtime = stream == nullptr ? &Runtime::get() : Runtime::owner_of_stream(stream);
    if (stream_runtime == nullptr || stream_runtime != runtime_) {
        err = "EXL3 PLE: stream belongs to another runtime";
        return false;
    }
    if (!key_->prepare_stream(stream, err) || !value_->prepare_stream(stream, err)) {
        err = "EXL3 PLE: preparing projection stream failed: " + err;
        return false;
    }
    return true;
}

bool Exl3Ple::release(std::string *err) noexcept {
    bool ok = true;
    std::string local_error;
    try {
        const OnDevice on(device_);
        if (runtime_ != nullptr) {
            try {
                runtime_->finish();
            } catch (const std::exception &e) {
                ok = false;
                local_error = std::string("EXL3 PLE: runtime synchronization failed: ") + e.what();
                Runtime::finish_all();
            }
        }
        key_.reset();
        value_.reset();
        if (device_ >= 0) {
            OnDevice on(device_);
            free_device(norm_key_);
            free_device(norm_query_);
            free_device(norm_conv_);
            free_device(conv1d_f16_);
            free_device(pre_key_);
            free_device(pre_value_);
        } else {
            free_device(norm_key_);
            free_device(norm_query_);
            free_device(norm_conv_);
            free_device(conv1d_f16_);
            free_device(pre_key_);
            free_device(pre_value_);
        }
    } catch (const std::exception &e) {
        ok = false;
        local_error = std::string("EXL3 PLE: teardown failed: ") + e.what();
    } catch (...) {
        ok = false;
        local_error = "EXL3 PLE: teardown failed with an unknown exception";
    }
    weights_ = {};
    runtime_ = nullptr;
    device_ = -1;
    if (err != nullptr)
        *err = std::move(local_error);
    return ok;
}

bool Exl3Ple::close(std::string &err) {
    err.clear();
    return release(&err);
}

void Exl3Ple::close() noexcept { (void)release(nullptr); }

} // namespace strata::core
