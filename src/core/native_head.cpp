// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
#include "strata/core/native_head.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/runtime.hpp"
#include "strata/artifact/exl3.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/raw_linear.hpp"
#include "strata/platform/memory.hpp"

#include "strata/core/gpu.hpp"
#include <climits>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>

namespace strata::core {

NativeHead::~NativeHead() {
    if (scratch_) strata::gpu::free(scratch_);
    if (weights_) strata::gpu::free(weights_);
}

void NativeEmbed::reset() {
    std::lock_guard<std::mutex> lock(copies_mutex_);
    for (void* c : copies_)
        if (c) strata::gpu::free(c);
    copies_.clear();
    if (host_) strata::gpu::free(host_);
    host_ = nullptr;
    dev_ = nullptr;
    bytes_ = 0;
    row_ = 0;
    n_embd_ = n_vocab_ = 0;
    type_ = -1;
    raw_ = false;
    raw_dtype_ = -1;
}

bool NativeHead::load(const std::string& path, int64_t n_in, int64_t n_out, std::string& err) {
    if (loaded()) { err = "native head is already loaded"; return false; }
    if (n_in <= 0 || n_out <= 0 || n_in > INT_MAX || n_out > INT_MAX || n_in % 256) {
        err = "native head requires positive int32 dimensions and whole 256-value rows";
        return false;
    }
    try {
        // The architecture is the metadata shard's; output.weight comes from whichever shard of `path`'s model
        // holds it (shard 2 of Unsloth's UD-Q4_K_XL, whose shard 1 holds no tensor; upstream 02cfe36).  GgufModel
        // refuses a missing shard and a duplicate across shards.
        const strata::GgufModel model = strata::GgufModel::open(path);
        err = strata::check_architecture(model.meta());
        if (!err.empty()) return false;
        size_t at = 0;
        const strata::TensorInfo* tensor = model.find("output.weight", &at);
        const strata::GgufFile& gguf = model.shard(at);
        if (!tensor || !strata::kernels::native_mmvq_supported((int) tensor->type) || tensor->shape.size() != 2 ||
            tensor->shape[0] != (uint64_t) n_in || tensor->shape[1] != (uint64_t) n_out) {
            err = "native head: expected a natively supported output.weight with the canonical head dimensions";
            return false;
        }
        const uint64_t bytes = strata::kernels::native_mmvq_weight_bytes((int) tensor->type, (int) n_in, (int) n_out);
        const uint64_t payload = gguf.file_size() - gguf.data_start();
        if (tensor->offset > payload || bytes > payload - tensor->offset) {
            err = "native head: truncated output.weight payload";
            return false;
        }
        void* weights = nullptr;
        void* scratch = nullptr;
        bool status = strata::gpu::alloc_device(&weights, bytes);
        if (status)
            status = strata::gpu::alloc_device(&scratch, strata::kernels::native_q8_1_bytes((int) n_in, 1));
        strata::platform::advise_willneed(gguf.tensor_data(*tensor), bytes);
        if (status)
            status = strata::gpu::copy(weights, gguf.tensor_data(*tensor), bytes);
        if (!status) {
            if (scratch) strata::gpu::free(scratch);
            if (weights) strata::gpu::free(weights);
            err = std::string("native head upload: ") + strata::gpu::last_error();
            return false;
        }
        int type = (int) tensor->type;
        if (type == 14 && strata::kernels::native_q6_k_rows_ok((int) n_in)) {   // aligned loads (native_mmvq.hpp)
            sycl::queue& q = strata::core::Runtime::get().compute();
            strata::kernels::native_q6_k_to_rows(weights, (int) n_in, (int) n_out, &q);
            q.wait();
            type = strata::kernels::kNativeQ6KRows;
        } else if (type == 8 && strata::kernels::native_q8_0_rows_ok((int) n_in)) {
            sycl::queue& q = strata::core::Runtime::get().compute();
            strata::kernels::native_q8_0_to_rows(weights, (int) n_in, (int) n_out, &q);
            q.wait();
            type = strata::kernels::kNativeQ8Rows;
        }
        weights_ = weights;
        scratch_ = scratch;
        bytes_ = bytes;
        n_in_ = (int) n_in;
        n_out_ = (int) n_out;
        type_ = type;
        return true;
    } catch (const std::exception& error) {
        err = std::string("native head: ") + error.what();
        return false;
    }
}

bool NativeHead::run(const float* mixed, float* logits, void* stream, std::string& err) const {
    if (!loaded() || !mixed || !logits || !stream) {
        err = "native head requires loaded weights, device buffers and an explicit stream";
        return false;
    }
    try {
        if (type_ == 13) {
            strata::kernels::native_q5_k_f32(weights_, mixed, scratch_, logits, n_in_, n_out_, 1, stream);
        } else {
            strata::kernels::native_quantize_q8_1(mixed, scratch_, n_in_, 1, stream);
            strata::kernels::native_mmvq(type_, weights_, scratch_, logits, n_in_, n_out_, 1, stream);
        }
    } catch (const std::exception& error) {
        err = std::string("native head launch: ") + error.what();
        return false;
    }
    return true;
}

// ================================ plan v0.3 P6: THE NATIVE EMBEDDING ================================

namespace {
const NativeEmbed* g_embed = nullptr;
}
void set_native_embed(const NativeEmbed* e) { g_embed = e; }
const NativeEmbed* native_embed() { return g_embed; }

NativeEmbed::~NativeEmbed() {
    reset();
}

const void* NativeEmbed::table_for(void* stream) const {
    const Runtime* table_rt = Runtime::owner(host_);
    const Runtime* run_rt = stream ? Runtime::owner_of_stream(stream) : &Runtime::get();
    if (!table_rt || !run_rt || table_rt == run_rt) return dev_;
    const int dev = run_rt->ordinal();
    std::lock_guard<std::mutex> lock(copies_mutex_);
    if (copies_.size() <= (size_t) dev) copies_.resize((size_t) dev + 1, nullptr);
    if (!copies_[(size_t) dev]) {
        const OnDevice on(dev);
        void* c = nullptr;
        if (!strata::gpu::alloc_host(&c, bytes_))
            throw std::runtime_error("native embedding: cannot pin a copy of " + std::to_string(bytes_ >> 20) +
                                     " MiB for GPU " + std::to_string(dev));
        std::memcpy(c, host_, bytes_);
        copies_[(size_t) dev] = c;
    }
    return copies_[(size_t) dev];
}

bool NativeEmbed::load(const std::string& path, int64_t n_embd, int64_t n_vocab, std::string& err) {
    if (host_ != nullptr) { err = "native embedding is already loaded"; return false; }
    try {
        // token_embd.weight from whichever shard of `path`'s model holds it (upstream 02cfe36)
        const strata::GgufModel model = strata::GgufModel::open(path);
        size_t at = 0;
        const strata::TensorInfo* t = model.find("token_embd.weight", &at);
        const strata::GgufFile& gguf = model.shard(at);
        if (!t || t->shape.size() != 2 || t->shape[0] != (uint64_t) n_embd || t->shape[1] != (uint64_t) n_vocab ||
            !strata::kernels::embed_type_supported((int) t->type) || n_embd % 256) {
            err = "native embedding: token_embd.weight is absent, of another shape, or of a type without a GPU "
                  "dequantizer";
            return false;
        }
        row_ = strata::kernels::iq_row_bytes((int) t->type, n_embd);
        bytes_ = (uint64_t) row_ * (uint64_t) n_vocab;
        // the table is copied out of the mapping below: a truncated shard must be an error, not a read past EOF
        if (!model.in_bounds(*t, at) || strata::tensor_payload_bytes(*t) != bytes_) {
            err = "native embedding: token_embd.weight's payload is truncated or not " + std::to_string(bytes_) +
                  " B (" + gguf.path() + ")";
            bytes_ = 0;
            return false;
        }
        strata::platform::advise_willneed(gguf.tensor_data(*t), bytes_);   // copied out of the mapping below
        if (!strata::gpu::alloc_host(&host_, bytes_)) {
            host_ = nullptr;
            err = "native embedding: cannot pin " + std::to_string(bytes_ >> 20) + " MiB";
            return false;
        }
        std::memcpy(host_, gguf.tensor_data(*t), bytes_);
        void* d = nullptr;
        if (!strata::gpu::device_pointer(&d, host_)) {
            err = "native embedding: no device alias for the mapped table";
            reset();
            return false;
        }
        dev_ = d;
        type_ = (int) t->type;
        n_embd_ = n_embd;
        n_vocab_ = n_vocab;
        raw_ = false;
        raw_dtype_ = -1;
        return true;
    } catch (const std::exception& e) {
        reset();
        err = std::string("native embedding: ") + e.what();
        return false;
    }
}

bool NativeEmbed::load_exl3(const strata::artifact::exl3::Archive& archive, const std::string& hf_name,
                            int64_t n_embd, int64_t n_vocab, std::string& err) {
    if (host_ != nullptr) { err = "native embedding is already loaded"; return false; }
    err.clear();
    try {
        if (hf_name.empty() || n_embd <= 0 || n_vocab <= 0) {
            err = "native embedding: EXL3 embedding requires positive dimensions and a tensor name";
            return false;
        }
        const auto* tensor = archive.find_tensor(hf_name + ".weight");
        if (tensor == nullptr || tensor->shape != std::vector<uint64_t>{static_cast<uint64_t>(n_vocab),
                                                                           static_cast<uint64_t>(n_embd)}) {
            err = "native embedding: EXL3 raw embedding has an absent or unexpected [vocab, embedding] shape";
            return false;
        }
        strata::kernels::raw::DType dtype;
        size_t element_bytes = 0;
        if (tensor->dtype == "F16") { dtype = strata::kernels::raw::DType::F16; element_bytes = 2; }
        else if (tensor->dtype == "BF16") { dtype = strata::kernels::raw::DType::BF16; element_bytes = 2; }
        else if (tensor->dtype == "F32") { dtype = strata::kernels::raw::DType::F32; element_bytes = 4; }
        else {
            err = "native embedding: EXL3 raw embedding dtype is not F16/BF16/F32";
            return false;
        }
        if ((uint64_t) n_embd > std::numeric_limits<size_t>::max() / element_bytes ||
            (uint64_t) n_vocab > std::numeric_limits<size_t>::max() / (static_cast<size_t>(n_embd) * element_bytes)) {
            err = "native embedding: EXL3 raw embedding size overflows the host address space";
            return false;
        }
        const size_t row_bytes = static_cast<size_t>(n_embd) * element_bytes;
        const uint64_t expected = static_cast<uint64_t>(row_bytes) * static_cast<uint64_t>(n_vocab);
        if (tensor->bytes != expected) {
            err = "native embedding: EXL3 raw embedding payload does not match its shape";
            return false;
        }
        std::vector<uint8_t> payload;
        archive.read_tensor(*tensor, payload);
        if (payload.size() != static_cast<size_t>(expected)) {
            err = "native embedding: EXL3 raw embedding payload is truncated";
            return false;
        }
        if (!strata::gpu::alloc_host(&host_, payload.size())) {
            err = "native embedding: cannot pin EXL3 raw embedding";
            return false;
        }
        std::memcpy(host_, payload.data(), payload.size());
        void* raw_dev = nullptr;
        if (!strata::gpu::device_pointer(&raw_dev, host_)) {
            err = "native embedding: EXL3 raw embedding has no device alias";
            reset();
            return false;
        }
        dev_ = raw_dev;
        bytes_ = expected;
        row_ = row_bytes;
        n_embd_ = n_embd;
        n_vocab_ = n_vocab;
        raw_dtype_ = static_cast<int>(dtype);
        // Preserve NativeEmbed::type()'s long-standing GGML type contract:
        // F32=0, F16=1, BF16=30. Raw dispatch uses raw_dtype_ separately.
        type_ = dtype == strata::kernels::raw::DType::F16 ? 1 :
                dtype == strata::kernels::raw::DType::BF16 ? 30 : 0;
        raw_ = true;
        return true;
    } catch (const std::exception& e) {
        reset();
        err = std::string("native embedding: ") + e.what();
        return false;
    }
}

void NativeEmbed::gather_dev(const int32_t* tokens, int64_t n_tok, float* out, void* stream) const {
    if (raw_) {
        strata::kernels::raw::embedding_gather(table_for(stream), static_cast<strata::kernels::raw::DType>(raw_dtype_),
                                               row_, tokens, n_tok, n_vocab_, n_embd_, out, n_embd_, stream);
    } else {
        strata::kernels::iq_embed_rows(type_, table_for(stream), row_, tokens, n_tok, n_embd_, out, stream);
    }
}

void NativeEmbed::gather_one(int64_t token, float* out, void* stream) const {
    if (token < 0 || token >= n_vocab_ || out == nullptr)
        throw std::invalid_argument("native embedding: token is outside the loaded vocabulary");
    if (raw_) {
        strata::kernels::raw::embedding_gather_one(table_for(stream), static_cast<strata::kernels::raw::DType>(raw_dtype_),
                                                   row_, token, n_vocab_, n_embd_, out, stream);
    } else {
        strata::kernels::iq_dequant_f32(type_, (const uint8_t*) table_for(stream) + (size_t) token * row_, n_embd_, out,
                                        stream);
    }
}

}  // namespace strata::core
