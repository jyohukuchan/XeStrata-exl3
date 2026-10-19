// SPDX-FileCopyrightText: 2025 Turboderp
// SPDX-FileCopyrightText: 2026 jyohukuchan
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// CPU EXL3 n-gram safetensors sidecar reader with blocking requested-row reads.
#include "strata/artifact/exl3_ngram.hpp"

#include "strata/kernels/f16_bits.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <cstdint>

namespace strata::artifact::exl3_ngram {
namespace {

using Header = strata::artifact::exl3::SafetensorsHeader;
using Tensor = strata::artifact::exl3::TensorDescriptor;
using Status = strata::kernels::cpu::exl3::Status;
constexpr int kRowDim = strata::kernels::exl3_ngram::kRowDim;

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error("EXL3 ngram: " + message); }

int64_t metadata_integer(const std::map<std::string, std::string>& metadata, const std::string& key) {
    const auto it = metadata.find(key);
    if (it == metadata.end() || it->second.empty()) fail("metadata " + key + " is missing");
    size_t used = 0;
    int64_t value;
    try {
        value = std::stoll(it->second, &used, 10);
    } catch (const std::exception&) {
        fail("metadata " + key + " is not an integer");
    }
    if (used != it->second.size()) fail("metadata " + key + " is not an integer");
    return value;
}

uint16_t u16(const uint8_t* p) { return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8); }

int64_t i64(const uint8_t* p) {
    uint64_t raw = 0;
    for (int i = 0; i < 8; ++i) raw |= static_cast<uint64_t>(p[i]) << (8 * i);
    int64_t value;
    std::memcpy(&value, &raw, sizeof(value));
    return value;
}

std::vector<uint16_t> read_u16(const Header& header, const Tensor& tensor) {
    std::vector<uint8_t> raw;
    strata::artifact::exl3::read_tensor(header, tensor, raw);
    if (raw.size() % 2 != 0) fail(tensor.name + " is not uint16 aligned");
    std::vector<uint16_t> result(raw.size() / 2);
    for (size_t i = 0; i < result.size(); ++i) result[i] = u16(raw.data() + i * 2);
    return result;
}

std::vector<int64_t> read_i64(const Header& header, const Tensor& tensor) {
    std::vector<uint8_t> raw;
    strata::artifact::exl3::read_tensor(header, tensor, raw);
    if (raw.size() % 8 != 0) fail(tensor.name + " is not int64 aligned");
    std::vector<int64_t> result(raw.size() / 8);
    for (size_t i = 0; i < result.size(); ++i) result[i] = i64(raw.data() + i * 8);
    return result;
}

std::vector<uint8_t> read_at(const Header& header, uint64_t absolute, size_t bytes) {
    if (absolute > header.file_bytes || bytes > header.file_bytes - absolute) fail("requested row span is outside the sidecar");
    std::vector<uint8_t> result(bytes);
    std::ifstream input(header.path, std::ios::binary);
    if (!input) fail("cannot open sidecar for row read");
    if (absolute > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) fail("row offset overflows streamoff");
    input.seekg(static_cast<std::streamoff>(absolute));
    if (!input || (!result.empty() && !input.read(reinterpret_cast<char*>(result.data()), static_cast<std::streamsize>(result.size()))))
        fail("short row read");
    return result;
}

const Tensor& require_vector(const std::map<std::string, Tensor>& tensors, const std::string& name,
                             const char* dtype, uint64_t elements) {
    const auto it = tensors.find(name);
    if (it == tensors.end() || it->second.dtype != dtype || it->second.shape.size() != 1 || it->second.shape[0] != elements)
        fail("invalid dtype or shape for " + name);
    return it->second;
}

int shard_number(const std::string& name, std::string& prefix) {
    const std::string marker = ".shard_";
    const size_t at = name.rfind(marker);
    if (at == std::string::npos || !name.ends_with(".trellis")) return -1;
    const size_t first = at + marker.size();
    const size_t last = name.size() - 8;
    if (first >= last) return -1;
    int value = 0;
    for (size_t i = first; i < last; ++i) {
        if (name[i] < '0' || name[i] > '9') return -1;
        const int digit = name[i] - '0';
        if (value > (std::numeric_limits<int>::max() - digit) / 10) return -1;
        value = value * 10 + digit;
    }
    prefix = name.substr(0, at);
    return value;
}

}  // namespace

struct Reader::Impl {
    Header header;
    std::string prefix;
    int bits = 0;
    size_t words = 0;
    int64_t row_count = 0;
    int64_t rows_per_shard = 0;
    std::vector<Tensor> shards;
    std::vector<RowSpan> spans;
    std::vector<int64_t> shard_rows;
    std::vector<int64_t> cumulative;
    std::vector<uint16_t> head_bias;
    std::vector<int64_t> head_offsets;
    std::vector<int64_t> head_vocab_sizes;
    std::vector<int64_t> layer_multipliers;
};

Reader::Reader(const std::filesystem::path& path) : impl_(std::make_unique<Impl>()) {
    impl_->header = strata::artifact::exl3::inspect_safetensors(path);
    const auto& metadata = impl_->header.metadata;
    const auto format = metadata.find("format");
    if (format == metadata.end() || format->second != "exl3_ngram_trellis") fail("format metadata is not exl3_ngram_trellis");
    const auto codebook = metadata.find("codebook");
    if (codebook == metadata.end() || codebook->second != "mul1") fail("n-gram sidecar requires mul1 codebook");
    const int64_t k = metadata_integer(metadata, "K");
    const int64_t row_dim = metadata_integer(metadata, "row_dim");
    if (k < 1 || k > 8) fail("metadata K is outside 1..8");
    if (row_dim != kRowDim) fail("metadata row_dim is not 160");
    impl_->bits = static_cast<int>(k);
    impl_->words = strata::kernels::exl3_ngram::words_per_row(impl_->bits);

    std::map<std::string, Tensor> tensors;
    for (const Tensor& tensor : impl_->header.tensors) {
        if (!tensors.emplace(tensor.name, tensor).second) fail("duplicate tensor " + tensor.name);
    }
    std::map<int, std::pair<std::string, Tensor>> shard_map;
    for (const auto& [name, tensor] : tensors) {
        std::string prefix;
        const int number = shard_number(name, prefix);
        if (number >= 0) {
            if (!shard_map.emplace(number, std::make_pair(prefix, tensor)).second) fail("duplicate shard number");
        }
    }
    if (shard_map.empty()) fail("no shard_N.trellis tensors");
    if (shard_map.begin()->first != 0) fail("shard numbering does not start at zero");
    for (int expected = 0; expected < static_cast<int>(shard_map.size()); ++expected)
        if (shard_map.find(expected) == shard_map.end()) fail("shard numbering is not contiguous");
    impl_->prefix = shard_map.begin()->second.first;
    int64_t nonfinal_rows = 0;
    for (const auto& [number, item] : shard_map) {
        (void) number;
        if (item.first != impl_->prefix) fail("shard prefixes differ");
        const Tensor& tensor = item.second;
        if (tensor.dtype != "I16" || tensor.shape.size() != 2 || tensor.shape[0] <= 0 || tensor.shape[1] != impl_->words)
            fail("invalid n-gram trellis shard shape for " + tensor.name);
        if (tensor.shape[0] > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) fail("shard row count overflows int64");
        const int64_t rows = static_cast<int64_t>(tensor.shape[0]);
        if (impl_->shards.size() + 1 < shard_map.size()) {
            if (nonfinal_rows == 0) nonfinal_rows = rows;
            if (rows != nonfinal_rows) fail("non-final shard row counts differ");
        }
        impl_->shards.push_back(tensor);
        impl_->spans.push_back({static_cast<uint64_t>(impl_->row_count), static_cast<uint64_t>(rows), tensor.data_offset_absolute});
        impl_->shard_rows.push_back(rows);
        if (impl_->row_count > std::numeric_limits<int64_t>::max() - rows) fail("row count overflows int64");
        impl_->row_count += rows;
    }
    impl_->rows_per_shard = impl_->shard_rows.front();
    impl_->cumulative.reserve(impl_->shard_rows.size());
    int64_t cumulative = 0;
    for (const int64_t rows : impl_->shard_rows) {
        cumulative += rows;
        impl_->cumulative.push_back(cumulative);
    }
    if (metadata.find("rows") != metadata.end() && metadata_integer(metadata, "rows") != impl_->row_count)
        fail("metadata rows disagrees with shard rows");
    if (metadata.find("shard_rows") != metadata.end() && metadata_integer(metadata, "shard_rows") != impl_->rows_per_shard)
        fail("metadata shard_rows disagrees with shard rows");

    const std::string head_bias_name = impl_->prefix + ".head_bias";
    const std::string offsets_name = impl_->prefix + ".head_offsets";
    const std::string sizes_name = impl_->prefix + ".head_vocab_sizes";
    const std::string multipliers_name = impl_->prefix + ".layer_multipliers";
    const auto head_bias_it = tensors.find(head_bias_name);
    if (head_bias_it == tensors.end() || head_bias_it->second.dtype != "F16" || head_bias_it->second.shape.size() != 2 ||
        head_bias_it->second.shape[1] != kRowDim)
        fail("invalid dtype or shape for " + head_bias_name);
    const Tensor& head_bias = head_bias_it->second;
    if (head_bias.shape[0] == 0 || head_bias.shape[0] > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
        fail("head_bias has no heads");
    impl_->head_bias = read_u16(impl_->header, head_bias);
    const Tensor& offsets = require_vector(tensors, offsets_name, "I64", head_bias.shape[0]);
    const Tensor& sizes = require_vector(tensors, sizes_name, "I64", head_bias.shape[0]);
    impl_->head_offsets = read_i64(impl_->header, offsets);
    impl_->head_vocab_sizes = read_i64(impl_->header, sizes);
    const auto multipliers_it = tensors.find(multipliers_name);
    if (multipliers_it == tensors.end() || multipliers_it->second.dtype != "I64" || multipliers_it->second.shape.size() != 1)
        fail("invalid dtype or shape for " + multipliers_name);
    impl_->layer_multipliers = read_i64(impl_->header, multipliers_it->second);
    // The final dimension is intentionally checked after lookup so variable
    // length I64 metadata can be represented without JSON in this API.
    if (impl_->layer_multipliers.empty()) fail("layer_multipliers is empty");
    if (impl_->layer_multipliers.size() < 2) fail("layer_multipliers is too short");
    for (size_t i = 0; i < impl_->head_offsets.size(); ++i) {
        const int64_t offset = impl_->head_offsets[i];
        const int64_t size = impl_->head_vocab_sizes[i];
        if (offset < 0 || size <= 0 || offset > impl_->row_count - size) fail("head range is outside the table");
        if (i == 0 && offset != 0) fail("head_offsets must start at zero");
        if (i > 0 && offset < impl_->head_offsets[i - 1]) fail("head_offsets are not sorted");
        if (i > 0 && offset < impl_->head_offsets[i - 1] + impl_->head_vocab_sizes[i - 1])
            fail("head ranges overlap");
    }
}

Reader::~Reader() = default;
Reader::Reader(Reader&&) noexcept = default;
Reader& Reader::operator=(Reader&&) noexcept = default;

const std::filesystem::path& Reader::path() const { return impl_->header.path; }
const std::string& Reader::prefix() const { return impl_->prefix; }
int Reader::bits() const { return impl_->bits; }
size_t Reader::words_per_row() const { return impl_->words; }
int64_t Reader::row_count() const { return impl_->row_count; }
int64_t Reader::rows_per_shard() const { return impl_->rows_per_shard; }
int64_t Reader::num_heads() const { return static_cast<int64_t>(impl_->head_offsets.size()); }
const std::vector<RowSpan>& Reader::row_spans() const { return impl_->spans; }
const std::vector<int64_t>& Reader::head_offsets() const { return impl_->head_offsets; }
const std::vector<int64_t>& Reader::head_vocab_sizes() const { return impl_->head_vocab_sizes; }
const std::vector<int64_t>& Reader::layer_multipliers() const { return impl_->layer_multipliers; }

void Reader::read_rows(const int64_t* row_ids, size_t count, std::vector<uint16_t>& packed) const {
    if (row_ids == nullptr && count > static_cast<size_t>(impl_->row_count)) fail("requested row count exceeds table");
    if (count > std::numeric_limits<size_t>::max() / impl_->words) fail("requested rows overflow size_t");
    std::vector<int64_t> ids(count);
    for (size_t i = 0; i < count; ++i) {
        ids[i] = row_ids == nullptr ? static_cast<int64_t>(i) : row_ids[i];
        if (ids[i] < 0 || ids[i] >= impl_->row_count) fail("row id is outside the table");
    }
    packed.assign(count * impl_->words, 0);
    struct Request { int shard; int64_t local; size_t output; };
    std::vector<Request> requests;
    requests.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        const auto shard_it = std::upper_bound(impl_->cumulative.begin(), impl_->cumulative.end(), ids[i]);
        const int shard = static_cast<int>(shard_it - impl_->cumulative.begin());
        const int64_t start = shard == 0 ? 0 : impl_->cumulative[static_cast<size_t>(shard - 1)];
        requests.push_back({shard, ids[i] - start, i});
    }
    std::sort(requests.begin(), requests.end(), [](const Request& a, const Request& b) {
        return a.shard != b.shard ? a.shard < b.shard : a.local < b.local;
    });
    size_t at = 0;
    while (at < requests.size()) {
        size_t end = at + 1;
        while (end < requests.size() && requests[end].shard == requests[at].shard &&
               requests[end].local == requests[end - 1].local + 1)
            ++end;
        const size_t run = end - at;
        const Tensor& shard = impl_->shards[static_cast<size_t>(requests[at].shard)];
        const uint64_t row_bytes = static_cast<uint64_t>(impl_->words) * sizeof(uint16_t);
        if (static_cast<uint64_t>(requests[at].local) > std::numeric_limits<uint64_t>::max() / row_bytes)
            fail("row read offset overflows uint64");
        const uint64_t row_offset = static_cast<uint64_t>(requests[at].local) * row_bytes;
        if (shard.data_offset > std::numeric_limits<uint64_t>::max() - row_offset) fail("row read offset overflows uint64");
        if (run > std::numeric_limits<size_t>::max() / static_cast<size_t>(row_bytes)) fail("row read size overflows size_t");
        const uint64_t relative = shard.data_offset + row_offset;
        Tensor read = shard;
        read.data_offset = relative;
        read.data_offset_absolute = impl_->header.data_start + relative;
        read.bytes = static_cast<uint64_t>(run * static_cast<size_t>(row_bytes));
        std::vector<uint8_t> raw = read_at(impl_->header, read.data_offset_absolute, static_cast<size_t>(read.bytes));
        for (size_t i = 0; i < run; ++i) {
            const uint8_t* source = raw.data() + i * impl_->words * sizeof(uint16_t);
            uint16_t* destination = packed.data() + requests[at + i].output * impl_->words;
            for (size_t word = 0; word < impl_->words; ++word) destination[word] = u16(source + word * 2);
        }
        at = end;
    }
}

Status Reader::decode_rows(const int64_t* row_ids, const int32_t* explicit_head_ids, size_t count, float* output,
                           size_t output_stride) const {
    std::vector<int64_t> ids(count);
    for (size_t i = 0; i < count; ++i) ids[i] = row_ids == nullptr ? static_cast<int64_t>(i) : row_ids[i];
    std::vector<int32_t> heads(count);
    if (explicit_head_ids == nullptr) {
        for (size_t i = 0; i < count; ++i) {
            int64_t head = -1;
            for (size_t h = 0; h < impl_->head_offsets.size(); ++h) {
                if (ids[i] >= impl_->head_offsets[h] && ids[i] < impl_->head_offsets[h] + impl_->head_vocab_sizes[h]) {
                    head = static_cast<int64_t>(h);
                    break;
                }
            }
            if (head < 0) return Status::invalid_argument;
            heads[i] = static_cast<int32_t>(head);
        }
    } else {
        std::copy(explicit_head_ids, explicit_head_ids + count, heads.begin());
    }
    std::vector<uint16_t> packed;
    read_rows(row_ids, count, packed);
    return decode_packed_rows(packed.data(), count, heads.data(), output, output_stride);
}

Status Reader::decode_packed_rows(const uint16_t* packed_rows, size_t count, const int32_t* explicit_head_ids,
                                  float* output, size_t output_stride) const {
    if (packed_rows != nullptr && (reinterpret_cast<uintptr_t>(packed_rows) % alignof(uint16_t)) != 0)
        return Status::unsupported_layout;
    if (count > static_cast<size_t>(std::numeric_limits<int64_t>::max())) return Status::size_overflow;
    if (impl_->words > std::numeric_limits<size_t>::max() / sizeof(uint16_t)) return Status::size_overflow;
    strata::kernels::exl3_ngram::View view{
        packed_rows, static_cast<int64_t>(count), static_cast<int64_t>(impl_->words), impl_->bits,
        impl_->head_bias.data(), static_cast<int64_t>(impl_->head_offsets.size())};
    return strata::kernels::cpu::exl3_ngram::decode_rows(view, nullptr, explicit_head_ids, count, output,
                                                          output_stride);
}

void Reader::lookup_ids(const int64_t* tokens, size_t token_count, int ngram_size, int heads_per_ngram,
                        int64_t eos_token_id, std::vector<int64_t>& output) const {
    if (tokens == nullptr || ngram_size < 2 || static_cast<size_t>(ngram_size) != impl_->layer_multipliers.size() ||
        heads_per_ngram <= 0 || static_cast<uint64_t>(heads_per_ngram) * static_cast<uint64_t>(ngram_size - 1) !=
                                   impl_->head_offsets.size() ||
        token_count < static_cast<size_t>(ngram_size - 1) ||
        token_count > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()) / sizeof(int64_t))
        fail("lookup metadata dimensions disagree");
    const size_t context = static_cast<size_t>(ngram_size - 1);
    const size_t output_rows = token_count - context;
    if (output_rows > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()) / sizeof(int64_t) /
                          impl_->head_offsets.size()) fail("lookup output exceeds host address range");
    output.resize(output_rows * impl_->head_offsets.size());
    int64_t previous_eos = -1;
    for (size_t candidate = 0; candidate < context; ++candidate)
        if (tokens[candidate] == eos_token_id) previous_eos = static_cast<int64_t>(candidate);
    for (size_t row = 0; row < output_rows; ++row) {
        const size_t position = row + context;
        uint64_t mixed = static_cast<uint64_t>(tokens[position]) * static_cast<uint64_t>(impl_->layer_multipliers[0]);
        for (int s = 1; s < ngram_size; ++s) {
            const int64_t source = static_cast<int64_t>(position) - s;
            const int64_t value = source >= previous_eos + 1 ? tokens[static_cast<size_t>(source)] : eos_token_id;
            mixed ^= static_cast<uint64_t>(value) * static_cast<uint64_t>(impl_->layer_multipliers[static_cast<size_t>(s)]);
            const size_t first = static_cast<size_t>(s - 1) * static_cast<size_t>(heads_per_ngram);
            for (int h = 0; h < heads_per_ngram; ++h) {
                const size_t head = first + static_cast<size_t>(h);
                int64_t signed_mixed;
                std::memcpy(&signed_mixed, &mixed, sizeof(signed_mixed));
                const int64_t modulo = signed_mixed % impl_->head_vocab_sizes[head];
                output[row * impl_->head_offsets.size() + head] = impl_->head_offsets[head] + (modulo < 0 ? modulo + impl_->head_vocab_sizes[head] : modulo);
            }
        }
        // The current token becomes a reset boundary for the next position,
        // after this position's hash has consumed only its preceding history.
        if (tokens[position] == eos_token_id) previous_eos = static_cast<int64_t>(position);
    }
}

}  // namespace strata::artifact::exl3_ngram
