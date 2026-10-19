// SPDX-FileCopyrightText: 2025 Turboderp
// SPDX-FileCopyrightText: 2026 jyohukuchan
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// CPU-side EXL3 safetensors archive descriptors and bounded materialization.
#include "strata/artifact/exl3.hpp"

#include "strata/kernels/f16_bits.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace strata::artifact::exl3 {
namespace {

// gitleaks:allow -- EXL3 marker constants are public format values, never secrets.
using Json = nlohmann::json;
using MatrixView = strata::kernels::exl3::MatrixView;

constexpr uint64_t kMaxHeaderBytes = 100ull * 1024ull * 1024ull;
constexpr const char* kKnownPatch = "mtp_hyper_connection_mixer_patch.safetensors";

struct ParsedFile {
    std::string name;
    std::vector<TensorDescriptor> tensors;
    std::string format = "safetensors";
};

struct Span {
    uint64_t begin = 0;
    uint64_t end = 0;
    std::string name;
};

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error("EXL3: " + message); }

bool is_integer(const Json& value) {
    return value.is_number_integer() || value.is_number_unsigned();
}

uint64_t json_u64(const Json& value, const std::string& label) {
    if (value.is_number_unsigned()) return value.get<uint64_t>();
    if (value.is_number_integer()) {
        const int64_t signed_value = value.get<int64_t>();
        if (signed_value >= 0) return static_cast<uint64_t>(signed_value);
    }
    fail(label + " must be a nonnegative integer");
}

std::string read_text(const std::filesystem::path& path, uint64_t max_bytes = kMaxHeaderBytes) {
    std::error_code ec;
    const uintmax_t file_size = std::filesystem::file_size(path, ec);
    if (ec) fail(path.filename().string() + ": cannot stat file: " + ec.message());
    if (file_size > max_bytes) fail(path.filename().string() + ": JSON file is too large");
    std::ifstream input(path, std::ios::binary);
    if (!input) fail(path.filename().string() + ": cannot open JSON");
    std::string result(static_cast<size_t>(file_size), '\0');
    if (!result.empty() && !input.read(result.data(), static_cast<std::streamsize>(result.size())))
        fail(path.filename().string() + ": truncated JSON");
    return result;
}

Json parse_json(const std::string& text, const std::string& label) {
    // nlohmann-json intentionally follows JavaScript's last-key-wins rule.  A
    // safetensors header cannot safely use that rule, so track object depth and
    // reject duplicate keys before the object is constructed.
    std::vector<std::set<std::string>> keys;
    Json::parser_callback_t callback = [&keys](int, Json::parse_event_t event, Json& parsed) {
        if (event == Json::parse_event_t::object_start) {
            keys.emplace_back();
        } else if (event == Json::parse_event_t::key) {
            const std::string key = parsed.get<std::string>();
            if (keys.empty() || !keys.back().insert(key).second) throw std::runtime_error("duplicate JSON key " + key);
        } else if (event == Json::parse_event_t::object_end) {
            if (!keys.empty()) keys.pop_back();
        }
        return true;
    };
    try {
        return Json::parse(text, callback, true, false);
    } catch (const std::exception& e) {
        fail(label + ": malformed JSON: " + e.what());
    }
}

std::string safe_basename(const Json& value, const std::string& label) {
    if (!value.is_string()) fail(label + ": shard filename must be a string");
    const std::string name = value.get<std::string>();
    if (name.empty() || name == "." || name == ".." || name.find('/') != std::string::npos ||
        name.find('\\') != std::string::npos || name.find('\0') != std::string::npos ||
        std::filesystem::path(name).filename().string() != name)
        fail(label + ": safetensors filename must be a plain basename: " + name);
    return name;
}

uint64_t dtype_bytes(const std::string& dtype) {
    static const std::map<std::string, uint64_t> sizes = {
        {"BOOL", 1}, {"U8", 1}, {"I8", 1}, {"U16", 2}, {"I16", 2}, {"U32", 4}, {"I32", 4},
        {"U64", 8}, {"I64", 8}, {"F8_E4M3", 1}, {"F8_E5M2", 1}, {"F8_E8M0", 1}, {"F16", 2},
        {"BF16", 2}, {"F32", 4}, {"F64", 8},
    };
    const auto it = sizes.find(dtype);
    return it == sizes.end() ? 0 : it->second;
}

bool checked_mul(uint64_t a, uint64_t b, uint64_t& result) {
    if (b != 0 && a > std::numeric_limits<uint64_t>::max() / b) return false;
    result = a * b;
    return true;
}

uint64_t shape_elements(const std::vector<uint64_t>& shape, const std::string& label) {
    uint64_t result = 1;
    for (const uint64_t dim : shape) {
        if (!checked_mul(result, dim, result)) fail(label + ": shape element count overflows uint64");
    }
    return result;
}

ParsedFile parse_safetensors(const std::filesystem::path& path) {
    std::error_code ec;
    const uintmax_t file_size = std::filesystem::file_size(path, ec);
    if (ec) fail(path.filename().string() + ": cannot stat safetensors file: " + ec.message());
    if (file_size < 8) fail(path.filename().string() + ": shorter than the safetensors header prefix");
    std::ifstream input(path, std::ios::binary);
    if (!input) fail(path.filename().string() + ": cannot open safetensors file");
    std::array<uint8_t, 8> prefix{};
    if (!input.read(reinterpret_cast<char*>(prefix.data()), prefix.size())) fail(path.filename().string() + ": truncated header prefix");
    uint64_t header_bytes = 0;
    for (int i = 0; i < 8; ++i) header_bytes |= static_cast<uint64_t>(prefix[static_cast<size_t>(i)]) << (8 * i);
    if (header_bytes == 0 || header_bytes > kMaxHeaderBytes || header_bytes > file_size - 8)
        fail(path.filename().string() + ": invalid header size");
    std::string raw_header(static_cast<size_t>(header_bytes), '\0');
    if (!input.read(raw_header.data(), static_cast<std::streamsize>(raw_header.size())))
        fail(path.filename().string() + ": truncated header");
    const Json header = parse_json(raw_header, path.filename().string());
    if (!header.is_object()) fail(path.filename().string() + ": header must be an object");

    ParsedFile result;
    result.name = path.filename().string();
    if (header.contains("__metadata__")) {
        const Json& metadata = header.at("__metadata__");
        if (!metadata.is_object()) fail(result.name + ": __metadata__ must be an object");
        for (auto it = metadata.begin(); it != metadata.end(); ++it)
            if (!it.value().is_string()) fail(result.name + ": __metadata__ values must be strings");
        const auto format = metadata.find("format");
        if (format != metadata.end()) result.format = format->get<std::string>();
    }

    const uint64_t data_start = 8 + header_bytes;
    std::vector<Span> spans;
    for (auto it = header.begin(); it != header.end(); ++it) {
        if (it.key() == "__metadata__") continue;
        const std::string& name = it.key();
        const Json& entry = it.value();
        if (!entry.is_object()) fail(result.name + ":" + name + ": tensor entry must be an object");
        const auto dtype_it = entry.find("dtype");
        if (dtype_it == entry.end() || !dtype_it->is_string()) fail(result.name + ":" + name + ": dtype is missing");
        const std::string dtype = dtype_it->get<std::string>();
        const uint64_t element_bytes = dtype_bytes(dtype);
        if (element_bytes == 0) fail(result.name + ":" + name + ": unsupported dtype " + dtype);
        const auto shape_it = entry.find("shape");
        if (shape_it == entry.end() || !shape_it->is_array()) fail(result.name + ":" + name + ": shape must be an array");
        std::vector<uint64_t> shape;
        shape.reserve(shape_it->size());
        for (size_t i = 0; i < shape_it->size(); ++i) shape.push_back(json_u64((*shape_it)[i], result.name + ":" + name + ": shape"));
        const auto offsets_it = entry.find("data_offsets");
        if (offsets_it == entry.end() || !offsets_it->is_array() || offsets_it->size() != 2)
            fail(result.name + ":" + name + ": data_offsets must contain two integers");
        const uint64_t begin = json_u64((*offsets_it)[0], result.name + ":" + name + ": data_offsets[0]");
        const uint64_t end = json_u64((*offsets_it)[1], result.name + ":" + name + ": data_offsets[1]");
        if (end < begin) fail(result.name + ":" + name + ": data_offsets are reversed");
        const uint64_t elements = shape_elements(shape, result.name + ":" + name);
        uint64_t expected = 0;
        if (!checked_mul(elements, element_bytes, expected) || end - begin != expected)
            fail(result.name + ":" + name + ": data_offsets span does not match dtype and shape");
        if (data_start > file_size || end > file_size - data_start)
            fail(result.name + ":" + name + ": tensor data extends past the file");
        TensorDescriptor tensor;
        tensor.name = name;
        tensor.dtype = dtype;
        tensor.shape = std::move(shape);
        tensor.file = result.name;
        tensor.data_offset = begin;
        tensor.data_offset_absolute = data_start + begin;
        tensor.bytes = expected;
        result.tensors.push_back(tensor);
        // An empty tensor occupies no bytes.  Do not let its zero-length span
        // participate in overlap ordering: it may legally share the offset
        // of the next nonempty tensor.
        if (begin != end) spans.push_back({begin, end, name});
    }
    std::sort(spans.begin(), spans.end(), [](const Span& a, const Span& b) { return a.begin < b.begin; });
    for (size_t i = 1; i < spans.size(); ++i)
        if (spans[i].begin < spans[i - 1].end)
            fail(result.name + ": tensor data overlaps: " + spans[i - 1].name + " and " + spans[i].name);
    return result;
}

uint32_t read_u32_le(const std::vector<uint8_t>& bytes) {
    if (bytes.size() != 4) fail("marker payload is not four bytes");
    return static_cast<uint32_t>(bytes[0]) | (static_cast<uint32_t>(bytes[1]) << 8) |
           (static_cast<uint32_t>(bytes[2]) << 16) | (static_cast<uint32_t>(bytes[3]) << 24);
}

uint16_t read_u16_le(const uint8_t* bytes) {
    return static_cast<uint16_t>(bytes[0]) | (static_cast<uint16_t>(bytes[1]) << 8);
}

float read_f32_le(const uint8_t* bytes) {
    uint32_t raw = static_cast<uint32_t>(bytes[0]) | (static_cast<uint32_t>(bytes[1]) << 8) |
                  (static_cast<uint32_t>(bytes[2]) << 16) | (static_cast<uint32_t>(bytes[3]) << 24);
    float value;
    std::memcpy(&value, &raw, sizeof(value));
    return value;
}

bool same_descriptor(const TensorDescriptor& a, const TensorDescriptor& b) {
    return a.name == b.name && a.dtype == b.dtype && a.shape == b.shape && a.file == b.file &&
           a.data_offset == b.data_offset && a.data_offset_absolute == b.data_offset_absolute && a.bytes == b.bytes;
}

}  // namespace

struct Archive::Impl {
    std::filesystem::path directory;
    std::string config;
    std::string quantization_config;
    std::vector<TensorDescriptor> indexed;
    std::vector<TensorDescriptor> effective;
    std::vector<LinearDescriptor> linears;
    std::vector<SidecarDescriptor> sidecars;
    std::unordered_map<std::string, size_t> effective_by_name;
    std::unordered_map<std::string, uint32_t> marker_values;
    std::vector<TensorDescriptor> all_known;
    bool has_total = false;
    uint64_t total = 0;
    uint64_t payload_bytes = 0;

    const TensorDescriptor* find(const std::string& name) const {
        const auto it = effective_by_name.find(name);
        return it == effective_by_name.end() ? nullptr : &effective[it->second];
    }

    bool known(const TensorDescriptor& tensor) const {
        return std::any_of(all_known.begin(), all_known.end(), [&](const TensorDescriptor& candidate) {
            return same_descriptor(candidate, tensor);
        });
    }
};

static void read_payload(const Archive::Impl& impl, const TensorDescriptor& tensor, std::vector<uint8_t>& output) {
    if (!impl.known(tensor)) fail(tensor.name + ": descriptor does not belong to this archive");
    const std::filesystem::path path = impl.directory / tensor.file;
    std::error_code ec;
    const uintmax_t file_size = std::filesystem::file_size(path, ec);
    if (ec || tensor.data_offset_absolute > file_size || tensor.bytes > file_size - tensor.data_offset_absolute)
        fail(tensor.name + ": payload is outside its file");
    if (tensor.bytes > std::numeric_limits<size_t>::max()) fail(tensor.name + ": payload is too large for this host");
    output.assign(static_cast<size_t>(tensor.bytes), 0);
    std::ifstream input(path, std::ios::binary);
    if (!input) fail(tensor.name + ": cannot open " + tensor.file);
    if (tensor.data_offset_absolute > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max()))
        fail(tensor.name + ": payload offset is too large");
    input.seekg(static_cast<std::streamoff>(tensor.data_offset_absolute));
    if (!input || (!output.empty() && !input.read(reinterpret_cast<char*>(output.data()), static_cast<std::streamsize>(output.size()))))
        fail(tensor.name + ": payload is truncated");
}

static uint32_t marker_for(const Archive::Impl& impl, const TensorDescriptor& tensor) {
    if ((tensor.dtype != "I32" && tensor.dtype != "U32") || !(tensor.shape.empty() || tensor.shape == std::vector<uint64_t>{1}) || tensor.bytes != 4)
        fail(tensor.name + ": codebook marker must be an I32/U32 scalar");
    const auto cached = impl.marker_values.find(tensor.name);
    if (cached != impl.marker_values.end()) return cached->second;
    std::vector<uint8_t> payload;
    read_payload(impl, tensor, payload);
    return read_u32_le(payload);
}

static void preload_markers(Archive::Impl& impl) {
    std::map<std::string, std::vector<const TensorDescriptor*>> by_file;
    for (const TensorDescriptor& tensor : impl.all_known) {
        const size_t dot = tensor.name.rfind('.');
        if (dot == std::string::npos) continue;
        const std::string suffix = tensor.name.substr(dot);
        if (suffix == ".mcg" || suffix == ".mul1") by_file[tensor.file].push_back(&tensor);
    }
    for (const auto& [file, tensors] : by_file) {
        std::ifstream input(impl.directory / file, std::ios::binary);
        if (!input) fail(file + ": cannot open marker shard");
        for (const TensorDescriptor* tensor : tensors) {
            if (tensor->data_offset_absolute > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max()))
                fail(tensor->name + ": marker offset is too large");
            std::array<uint8_t, 4> payload{};
            input.seekg(static_cast<std::streamoff>(tensor->data_offset_absolute));
            if (!input.read(reinterpret_cast<char*>(payload.data()), payload.size())) fail(tensor->name + ": marker is truncated");
            impl.marker_values.emplace(tensor->name, static_cast<uint32_t>(payload[0]) |
                                                       (static_cast<uint32_t>(payload[1]) << 8) |
                                                       (static_cast<uint32_t>(payload[2]) << 16) |
                                                       (static_cast<uint32_t>(payload[3]) << 24));
        }
    }
}

static void append_effective(Archive::Impl& impl, const TensorDescriptor& tensor) {
    auto it = impl.effective_by_name.find(tensor.name);
    if (it == impl.effective_by_name.end()) {
        impl.effective_by_name.emplace(tensor.name, impl.effective.size());
        impl.effective.push_back(tensor);
    } else {
        impl.effective[it->second] = tensor;
    }
}

static void build_linears(Archive::Impl& impl) {
    std::map<std::string, TensorDescriptor> by_name;
    for (const TensorDescriptor& tensor : impl.effective) by_name.emplace(tensor.name, tensor);
    for (const auto& [name, trellis] : by_name) {
        constexpr const char* suffix = ".trellis";
        if (name.size() <= std::strlen(suffix) || name.compare(name.size() - std::strlen(suffix), std::strlen(suffix), suffix) != 0)
            continue;
        if (trellis.dtype != "I16" || trellis.shape.size() != 3) fail(name + ": trellis must be a 3-D I16 tensor");
        if (trellis.shape[0] == 0 || trellis.shape[1] == 0 || trellis.shape[2] == 0 || trellis.shape[2] % 16 != 0)
            fail(name + ": trellis dimensions are invalid");
        const uint64_t bits = trellis.shape[2] / 16;
        if (bits < 1 || bits > 8) fail(name + ": trellis bit width is outside 1..8");
        uint64_t k = 0, n = 0;
        if (!checked_mul(trellis.shape[0], 16, k) || !checked_mul(trellis.shape[1], 16, n) ||
            k > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) || n > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
            fail(name + ": matrix dimensions overflow");
        const std::string base = name.substr(0, name.size() - std::strlen(suffix));
        auto get = [&](const char* ext) -> std::optional<TensorDescriptor> {
            const auto it = by_name.find(base + ext);
            return it == by_name.end() ? std::nullopt : std::optional<TensorDescriptor>(it->second);
        };
        LinearDescriptor linear;
        linear.name = base;
        linear.trellis = trellis;
        linear.k = static_cast<int64_t>(k);
        linear.n = static_cast<int64_t>(n);
        linear.bits = static_cast<int>(bits);
        linear.suh = get(".suh");  // codespell:ignore suh
        linear.svh = get(".svh");
        linear.su = get(".su");
        linear.sv = get(".sv");
        linear.bias = get(".bias");
        if ((linear.suh.has_value() && linear.su.has_value()) || (linear.svh.has_value() && linear.sv.has_value()))
            fail(base + ": both packed and unpacked scale companions are present");
        if (!linear.suh.has_value() && !linear.su.has_value()) fail(base + ": missing suh or su");  // codespell:ignore suh
        if (!linear.svh.has_value() && !linear.sv.has_value()) fail(base + ": missing svh or sv");
        if (linear.suh.has_value() && (linear.suh->dtype != "F16" || linear.suh->shape != std::vector<uint64_t>{k}))
            fail(linear.suh->name + ": suh must be F16 with shape [k]");  // codespell:ignore suh
        if (linear.svh.has_value() && (linear.svh->dtype != "F16" || linear.svh->shape != std::vector<uint64_t>{n}))
            fail(linear.svh->name + ": svh must be F16 with shape [n]");
        if (linear.su.has_value() && ((linear.su->dtype != "I16" && linear.su->dtype != "U16") || linear.su->shape != std::vector<uint64_t>{k / 16}))
            fail(linear.su->name + ": su must be I16/U16 with shape [k/16]");
        if (linear.sv.has_value() && ((linear.sv->dtype != "I16" && linear.sv->dtype != "U16") || linear.sv->shape != std::vector<uint64_t>{n / 16}))
            fail(linear.sv->name + ": sv must be I16/U16 with shape [n/16]");
        if (linear.bias.has_value() && ((linear.bias->dtype != "F16" && linear.bias->dtype != "F32") || linear.bias->shape != std::vector<uint64_t>{n}))
            fail(linear.bias->name + ": bias must be F16/F32 with shape [n]");

        const auto mcg = by_name.find(base + ".mcg");
        const auto mul1 = by_name.find(base + ".mul1");
        if (mcg != by_name.end() && mul1 != by_name.end()) fail(base + ": mcg and mul1 markers are mutually exclusive");
        if (mcg != by_name.end()) {
            linear.marker = marker_for(impl, mcg->second);
            if (linear.marker != 0xCBAC1FEDu) fail(mcg->second.name + ": unexpected mcg marker value");
            linear.codebook = 1;
        } else if (mul1 != by_name.end()) {
            linear.marker = marker_for(impl, mul1->second);
            if (linear.marker != 0x83DCD12Du) fail(mul1->second.name + ": unexpected mul1 marker value");
            linear.codebook = 2;
        }
        impl.linears.push_back(std::move(linear));
    }
}

static void load_main_files(Archive::Impl& impl, const std::filesystem::path& index_path) {
    std::vector<std::string> main_names;
    std::map<std::string, std::string> weight_map;
    if (!index_path.empty()) {
        const Json index = parse_json(read_text(index_path), index_path.filename().string());
        const auto map_it = index.find("weight_map");
        if (map_it == index.end() || !map_it->is_object() || map_it->empty()) fail("index: weight_map must be a nonempty object");
        for (auto it = map_it->begin(); it != map_it->end(); ++it) weight_map.emplace(it.key(), safe_basename(it.value(), "index"));
        std::set<std::string> unique;
        for (const auto& [tensor_name, file_name] : weight_map) {
            (void) tensor_name;
            if (unique.insert(file_name).second) main_names.push_back(file_name);
        }
        const auto metadata = index.find("metadata");
        if (metadata != index.end() && metadata->is_object()) {
            const auto total = metadata->find("total_size");
            if (total != metadata->end() && is_integer(*total)) {
                impl.has_total = true;
                impl.total = json_u64(*total, "index metadata.total_size");
            }
        }
    } else {
        const auto model = impl.directory / "model.safetensors";
        if (std::filesystem::is_regular_file(model)) {
            main_names.push_back(model.filename().string());
        } else {
            std::vector<std::string> candidates;
            for (const auto& entry : std::filesystem::directory_iterator(impl.directory))
                if (entry.is_regular_file() && entry.path().extension() == ".safetensors" && entry.path().filename().string() != kKnownPatch)
                    candidates.push_back(entry.path().filename().string());
            if (candidates.size() != 1) fail("archive needs model.safetensors or an index with exactly one main shard");
            main_names = std::move(candidates);
        }
    }
    std::set<std::string> indexed_names;
    for (const std::string& name : main_names) {
        const auto path = impl.directory / name;
        if (!std::filesystem::is_regular_file(path)) fail("missing main shard " + name);
        ParsedFile parsed = parse_safetensors(path);
        for (const TensorDescriptor& tensor : parsed.tensors) {
            if (!indexed_names.insert(tensor.name).second)
                fail("duplicate tensor across main shards: " + tensor.name);
            if (std::numeric_limits<uint64_t>::max() - impl.payload_bytes < tensor.bytes)
                fail("indexed tensor payload byte count overflows uint64");
            impl.indexed.push_back(tensor);
            impl.payload_bytes += tensor.bytes;
        }
    }
    if (!weight_map.empty()) {
        std::map<std::string, std::string> actual;
        for (const TensorDescriptor& tensor : impl.indexed) actual.emplace(tensor.name, tensor.file);
        if (actual.size() != weight_map.size()) fail("index weight_map/header tensor mismatch");
        for (const auto& [name, file] : weight_map) {
            const auto it = actual.find(name);
            if (it == actual.end() || it->second != file) fail("index weight_map does not match shard headers for " + name);
        }
    }
}

static void load_sidecars(Archive::Impl& impl, const std::set<std::string>& main_names) {
    for (const auto& entry : std::filesystem::directory_iterator(impl.directory)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".safetensors") continue;
        const std::string name = entry.path().filename().string();
        if (main_names.count(name) != 0) continue;
        ParsedFile parsed = parse_safetensors(entry.path());
        SidecarDescriptor sidecar;
        sidecar.file = name;
        sidecar.format = parsed.format;
        if (parsed.format == "exl3_ngram_trellis") sidecar.role = SidecarRole::specialized_ngram_codec;
        else if (name == kKnownPatch) {
            sidecar.role = SidecarRole::known_override_patch;
            sidecar.overrides_base = true;
        }
        sidecar.tensors = parsed.tensors;
        for (const TensorDescriptor& tensor : sidecar.tensors) {
            const TensorDescriptor* existing = impl.find(tensor.name);
            if (existing != nullptr && sidecar.role != SidecarRole::known_override_patch)
                fail("sidecar tensor duplicates indexed tensor: " + tensor.name);
            // Specialized n-gram tensors and unindexed sidecars stay outside
            // the linear descriptor map.  Only the explicitly known patch is
            // merged into the effective base view.
            if (sidecar.role == SidecarRole::known_override_patch) append_effective(impl, tensor);
            impl.all_known.push_back(tensor);
        }
        impl.sidecars.push_back(std::move(sidecar));
    }
}

Archive::Archive(const std::filesystem::path& directory) : impl_(std::make_unique<Impl>()) {
    impl_->directory = std::filesystem::absolute(directory).lexically_normal();
    if (!std::filesystem::is_directory(impl_->directory)) fail("archive directory does not exist: " + impl_->directory.string());
    const auto config_path = impl_->directory / "config.json";
    if (!std::filesystem::is_regular_file(config_path)) fail("config.json is required");
    impl_->config = read_text(config_path);
    if (!parse_json(impl_->config, "config.json").is_object()) fail("config.json: top level must be an object");
    const auto quant_path = impl_->directory / "quantization_config.json";
    if (std::filesystem::is_regular_file(quant_path)) {
        // This optional file can contain a model-wide per-tensor table tens of
        // megabytes larger than the safetensors index.  Keep it as opaque
        // caller-visible text; the loader does not consume its values to
        // identify EXL3 tensors, so parsing it here would add a needless
        // model-sized JSON allocation to archive open.
        impl_->quantization_config = read_text(quant_path);
    }
    const auto index_path = impl_->directory / "model.safetensors.index.json";
    load_main_files(*impl_, std::filesystem::is_regular_file(index_path) ? index_path : std::filesystem::path{});
    std::set<std::string> main_names;
    for (const TensorDescriptor& tensor : impl_->indexed) main_names.insert(tensor.file);
    impl_->all_known = impl_->indexed;
    for (const TensorDescriptor& tensor : impl_->indexed) append_effective(*impl_, tensor);
    load_sidecars(*impl_, main_names);
    std::sort(impl_->effective.begin(), impl_->effective.end(), [](const auto& a, const auto& b) { return a.name < b.name; });
    impl_->effective_by_name.clear();
    for (size_t i = 0; i < impl_->effective.size(); ++i) impl_->effective_by_name.emplace(impl_->effective[i].name, i);
    preload_markers(*impl_);
    build_linears(*impl_);
}

Archive::~Archive() = default;
Archive::Archive(Archive&&) noexcept = default;
Archive& Archive::operator=(Archive&&) noexcept = default;

const std::filesystem::path& Archive::directory() const { return impl_->directory; }
const std::string& Archive::config_json() const { return impl_->config; }
const std::string& Archive::quantization_config_json() const { return impl_->quantization_config; }
const std::vector<TensorDescriptor>& Archive::indexed_tensors() const { return impl_->indexed; }
const std::vector<TensorDescriptor>& Archive::tensors() const { return impl_->effective; }
const std::vector<LinearDescriptor>& Archive::linears() const { return impl_->linears; }
const std::vector<SidecarDescriptor>& Archive::sidecars() const { return impl_->sidecars; }
const TensorDescriptor* Archive::find_tensor(const std::string& name) const { return impl_->find(name); }
bool Archive::has_declared_total_size() const { return impl_->has_total; }
uint64_t Archive::declared_total_size() const { return impl_->total; }
uint64_t Archive::indexed_payload_bytes() const { return impl_->payload_bytes; }

void Archive::read_tensor(const TensorDescriptor& tensor, std::vector<uint8_t>& bytes) const { read_payload(*impl_, tensor, bytes); }

void Archive::materialize(const LinearDescriptor& linear, MaterializedMatrix& output) const {
    const LinearDescriptor* canonical = nullptr;
    for (const LinearDescriptor& candidate : impl_->linears) {
        if (candidate.name == linear.name) {
            canonical = &candidate;
            break;
        }
    }
    if (canonical == nullptr) fail(linear.name + ": linear descriptor is not from this archive");
    const auto same_optional = [](const std::optional<TensorDescriptor>& a,
                                  const std::optional<TensorDescriptor>& b) {
        if (a.has_value() != b.has_value()) return false;
        return !a.has_value() || same_descriptor(*a, *b);
    };
    if (linear.k != canonical->k || linear.n != canonical->n || linear.bits != canonical->bits ||
        linear.codebook != canonical->codebook || linear.marker != canonical->marker ||
        !same_descriptor(linear.trellis, canonical->trellis) || !same_optional(linear.suh, canonical->suh) ||
        !same_optional(linear.svh, canonical->svh) || !same_optional(linear.su, canonical->su) ||
        !same_optional(linear.sv, canonical->sv) || !same_optional(linear.bias, canonical->bias)
    ) fail(linear.name + ": linear descriptor does not match the archive's canonical descriptor");
    const LinearDescriptor& source = *canonical;
    MaterializedMatrix result;
    result.k = source.k;
    result.n = source.n;
    result.bits = source.bits;
    result.codebook = source.codebook;
    auto read_u16_vector = [&](const TensorDescriptor& tensor, std::vector<uint16_t>& destination) {
        std::vector<uint8_t> raw;
        read_payload(*impl_, tensor, raw);
        if (raw.size() % 2 != 0) fail(tensor.name + ": F16/I16 payload is not word aligned");
        destination.resize(raw.size() / 2);
        for (size_t i = 0; i < destination.size(); ++i) destination[i] = read_u16_le(raw.data() + i * 2);
    };
    read_u16_vector(source.trellis, result.trellis);
    auto scales = [&](const std::optional<TensorDescriptor>& unpacked, const std::optional<TensorDescriptor>& packed,
                      size_t expected, std::vector<uint16_t>& destination, const std::string& label) {
        if (unpacked.has_value()) {
            read_u16_vector(*unpacked, destination);
            if (destination.size() != expected) fail(label + ": wrong unpacked scale length");
        } else if (packed.has_value()) {
            std::vector<uint16_t> words;
            read_u16_vector(*packed, words);
            if (words.size() * 16 != expected) fail(label + ": wrong packed scale length");
            destination.resize(expected);
            for (size_t i = 0; i < words.size(); ++i)
                for (int bit = 0; bit < 16; ++bit)
                    destination[i * 16 + static_cast<size_t>(bit)] = (words[i] & (static_cast<uint16_t>(1u) << bit)) ? 0x3c00u : 0xbc00u;
        } else {
            fail(label + ": scale companion is missing");
        }
    };
    scales(source.suh, source.su, static_cast<size_t>(source.k), result.suh, source.name + ".suh");  // codespell:ignore suh
    scales(source.svh, source.sv, static_cast<size_t>(source.n), result.svh, source.name + ".svh");
    if (source.bias.has_value()) {
        if (source.bias->dtype == "F16") {
            read_u16_vector(*source.bias, result.bias);
        } else if (source.bias->dtype == "F32") {
            std::vector<uint8_t> raw;
            read_payload(*impl_, *source.bias, raw);
            if (raw.size() != static_cast<size_t>(source.n) * 4) fail(source.bias->name + ": wrong F32 bias length");
            result.bias.resize(static_cast<size_t>(source.n));
            for (size_t i = 0; i < result.bias.size(); ++i)
                result.bias[i] = strata::kernels::f16_from_f32(read_f32_le(raw.data() + i * 4));
        }
        if (result.bias.size() != static_cast<size_t>(source.n)) fail(source.bias->name + ": wrong bias length");
    }
    output = std::move(result);
}

MatrixView MaterializedMatrix::view() const {
    MatrixView result;
    result.trellis = trellis.empty() ? nullptr : trellis.data();
    result.suh = suh.empty() ? nullptr : suh.data();  // codespell:ignore suh
    result.svh = svh.empty() ? nullptr : svh.data();
    result.bias = bias.empty() ? nullptr : bias.data();
    result.k = k;
    result.n = n;
    result.bits = bits;
    result.codebook = codebook;
    return result;
}

}  // namespace strata::artifact::exl3
