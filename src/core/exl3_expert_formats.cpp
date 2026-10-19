// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "strata/core/exl3_expert_formats.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>

namespace strata::core {
namespace {

constexpr uint64_t kFnvOffset = 1469598103934665603ull;
constexpr uint64_t kFnvPrime = 1099511628211ull;

void hash_u64(uint64_t& h, uint64_t value) {
    for (int i = 0; i < 8; ++i) {
        h ^= (value >> (8 * i)) & 0xFFu;
        h *= kFnvPrime;
    }
}

void hash_component(uint64_t& h, const strata::kernels::exl3::ExpertComponentLayout& c) {
    hash_u64(h, c.offset);
    hash_u64(h, c.bytes);
}

void hash_projection(uint64_t& h, const strata::kernels::exl3::ExpertProjectionLayout& p) {
    hash_u64(h, static_cast<uint64_t>(p.storage));
    hash_u64(h, (uint64_t) p.k);
    hash_u64(h, (uint64_t) p.n);
    hash_u64(h, (uint64_t) p.bits);
    hash_u64(h, (uint64_t) p.codebook);
    hash_component(h, p.trellis);
    hash_component(h, p.suh);
    hash_component(h, p.svh);
    hash_component(h, p.bias);
    hash_u64(h, static_cast<uint64_t>(p.raw_dtype));
    hash_u64(h, static_cast<uint64_t>(p.raw_bias_dtype));
    hash_component(h, p.raw_weights);
    hash_component(h, p.raw_bias);
}

bool valid_raw_dtype(strata::kernels::raw::DType dtype) {
    return dtype == strata::kernels::raw::DType::F16 || dtype == strata::kernels::raw::DType::BF16 ||
           dtype == strata::kernels::raw::DType::F32;
}

bool component_span(const strata::kernels::exl3::ExpertComponentLayout& c, uint64_t total, uint32_t alignment,
                    uint64_t expected, const char* name, std::vector<std::pair<uint64_t, uint64_t>>& spans,
                    std::string& error) {
    if (c.bytes != 0 && c.bytes != expected) {
        error = std::string("EXL3 ") + name + " has unexpected byte length";
        return false;
    }
    if (c.bytes == 0) return true;
    if (alignment == 0 || c.offset % alignment != 0 || c.offset > total || c.bytes > total - c.offset) {
        error = std::string("EXL3 ") + name + " span is outside total bytes or misaligned";
        return false;
    }
    spans.emplace_back(c.offset, c.offset + c.bytes);
    return true;
}

bool projection_valid(const strata::kernels::exl3::ExpertProjectionLayout& p, int64_t expect_k, int64_t expect_n,
                      const char* role, uint32_t alignment, uint64_t total,
                      std::vector<std::pair<uint64_t, uint64_t>>& spans, std::string& error) {
    if (p.k != expect_k || p.n != expect_n || p.k <= 0 || p.n <= 0) {
        error = std::string("EXL3 ") + role + " geometry is invalid";
        return false;
    }
    const uint64_t k = (uint64_t) p.k, n = (uint64_t) p.n;
    if (p.storage == strata::kernels::exl3::ProjectionStorage::packed_trellis) {
        if (p.bits < 1 || p.bits > 8 || p.codebook < 0 || p.codebook > 2 || p.raw_weights.bytes != 0 ||
            p.raw_bias.bytes != 0) {
            error = std::string("EXL3 ") + role + " packed geometry or codec is invalid";
            return false;
        }
        const uint64_t bits = (uint64_t) p.bits;
        if (k > std::numeric_limits<uint64_t>::max() / n) {
            error = std::string("EXL3 ") + role + " trellis element count overflows";
            return false;
        }
        const uint64_t elements = k * n;
        if (elements > std::numeric_limits<uint64_t>::max() / bits) {
            error = std::string("EXL3 ") + role + " trellis byte count overflows";
            return false;
        }
        const uint64_t trellis = elements * bits / 8;
        if (p.trellis.bytes == 0 || p.suh.bytes == 0 || p.svh.bytes == 0) {
            error = std::string("EXL3 ") + role + " must provide trellis, suh and svh spans";
            return false;
        }
        if (!component_span(p.trellis, total, alignment, trellis, (std::string(role) + ".trellis").c_str(), spans, error) ||
            !component_span(p.suh, total, alignment, (uint64_t) p.k * 2, (std::string(role) + ".suh").c_str(), spans, error) ||
            !component_span(p.svh, total, alignment, (uint64_t) p.n * 2, (std::string(role) + ".svh").c_str(), spans, error) ||
            !component_span(p.bias, total, alignment, (uint64_t) p.n * 2, (std::string(role) + ".bias").c_str(), spans, error))
            return false;
        if ((p.suh.bytes == 0) != (p.svh.bytes == 0)) {
            error = std::string("EXL3 ") + role + " must provide suh and svh together";
            return false;
        }
        return true;
    }
    if (p.storage != strata::kernels::exl3::ProjectionStorage::raw || p.bits != 0 || p.codebook != 0 ||
        !valid_raw_dtype(p.raw_dtype) || !valid_raw_dtype(p.raw_bias_dtype) || p.trellis.bytes != 0 ||
        p.suh.bytes != 0 || p.svh.bytes != 0 || p.bias.bytes != 0) {
        error = std::string("EXL3 ") + role + " raw discriminator or dtype is invalid";
        return false;
    }
    if (k > std::numeric_limits<uint64_t>::max() / n) {
        error = std::string("EXL3 ") + role + " raw element count overflows";
        return false;
    }
    const uint64_t elements = k * n;
    const uint64_t weight_size = (uint64_t) strata::kernels::raw::dtype_bytes(p.raw_dtype);
    if (elements > std::numeric_limits<uint64_t>::max() / weight_size || p.raw_weights.bytes == 0) {
        error = std::string("EXL3 ") + role + " raw weight span is invalid";
        return false;
    }
    if (p.raw_weights.offset % weight_size != 0) {
        error = std::string("EXL3 ") + role + " raw weight span is misaligned";
        return false;
    }
    const uint64_t weight_bytes = elements * weight_size;
    const uint64_t bias_size = (uint64_t) strata::kernels::raw::dtype_bytes(p.raw_bias_dtype);
    if (n > std::numeric_limits<uint64_t>::max() / bias_size) {
        error = std::string("EXL3 ") + role + " raw bias span overflows";
        return false;
    }
    const uint64_t bias_bytes = p.raw_bias.bytes == 0 ? 0 : n * bias_size;
    if (p.raw_bias.bytes != 0 && p.raw_bias.offset % bias_size != 0) {
        error = std::string("EXL3 ") + role + " raw bias span is misaligned";
        return false;
    }
    if (!component_span(p.raw_weights, total, alignment, weight_bytes, (std::string(role) + ".raw_weights").c_str(), spans, error) ||
        !component_span(p.raw_bias, total, alignment, bias_bytes, (std::string(role) + ".raw_bias").c_str(), spans, error))
        return false;
    return true;
}

}  // namespace

bool exl3_expert_format_equal(const Exl3ExpertFormatDesc& a, const Exl3ExpertFormatDesc& b) {
    if (a.total_bytes != b.total_bytes || a.alignment != b.alignment) return false;
    const auto equal_projection = [](const auto& x, const auto& y) {
        return x.storage == y.storage && x.k == y.k && x.n == y.n && x.bits == y.bits && x.codebook == y.codebook &&
               x.trellis.offset == y.trellis.offset && x.trellis.bytes == y.trellis.bytes &&
               x.suh.offset == y.suh.offset && x.suh.bytes == y.suh.bytes &&
               x.svh.offset == y.svh.offset && x.svh.bytes == y.svh.bytes &&
               x.bias.offset == y.bias.offset && x.bias.bytes == y.bias.bytes && x.raw_dtype == y.raw_dtype &&
               x.raw_bias_dtype == y.raw_bias_dtype && x.raw_weights.offset == y.raw_weights.offset &&
               x.raw_weights.bytes == y.raw_weights.bytes && x.raw_bias.offset == y.raw_bias.offset &&
               x.raw_bias.bytes == y.raw_bias.bytes;
    };
    return equal_projection(a.gate, b.gate) && equal_projection(a.up, b.up) && equal_projection(a.down, b.down);
}

bool validate_exl3_expert_format(const Exl3ExpertFormatDesc& desc, std::string& error) {
    error.clear();
    const int64_t hidden = desc.gate.k, ff = desc.gate.n;
    if (hidden <= 0 || ff <= 0) {
        error = "EXL3 hidden/ff geometry must be positive";
        return false;
    }
    if (desc.alignment == 0 || desc.alignment > 4096 || (desc.alignment & (desc.alignment - 1)) != 0) {
        error = "EXL3 alignment must be a power of two in 1..4096";
        return false;
    }
    if (desc.total_bytes == 0 || desc.total_bytes > (uint64_t) std::numeric_limits<std::ptrdiff_t>::max()) {
        error = "EXL3 total byte size is zero or exceeds PTRDIFF_MAX";
        return false;
    }
    std::vector<std::pair<uint64_t, uint64_t>> spans;
    if (!projection_valid(desc.gate, hidden, ff, "gate", desc.alignment, desc.total_bytes, spans, error) ||
        !projection_valid(desc.up, hidden, ff, "up", desc.alignment, desc.total_bytes, spans, error) ||
        !projection_valid(desc.down, ff, hidden, "down", desc.alignment, desc.total_bytes, spans, error))
        return false;
    const auto packed_aligned = [](const auto& p) {
        return p.storage != strata::kernels::exl3::ProjectionStorage::packed_trellis ||
               (p.k % 128 == 0 && p.n % 128 == 0);
    };
    if (!packed_aligned(desc.gate) || !packed_aligned(desc.up) || !packed_aligned(desc.down)) {
        error = "EXL3 packed projection geometry must be a multiple of 128";
        return false;
    }
    std::sort(spans.begin(), spans.end());
    for (size_t i = 1; i < spans.size(); ++i)
        if (spans[i - 1].second > spans[i].first) {
            error = "EXL3 component spans overlap";
            return false;
        }
    return true;
}

Exl3ExpertFormatID exl3_expert_format_id(const Exl3ExpertFormatDesc& desc) {
    std::string error;
    if (!validate_exl3_expert_format(desc, error)) return 0;
    uint64_t h = kFnvOffset;
    hash_u64(h, desc.total_bytes);
    hash_u64(h, desc.alignment);
    hash_projection(h, desc.gate);
    hash_projection(h, desc.up);
    hash_projection(h, desc.down);
    return h == 0 ? 1 : h;
}

Exl3ExpertFormatID Exl3ExpertFormatRegistry::register_format(const Exl3ExpertFormatDesc& desc, std::string& error) {
    if (!validate_exl3_expert_format(desc, error)) return 0;
    const Exl3ExpertFormatID id = exl3_expert_format_id(desc);
    for (const Entry& entry : entries_) {
        if (entry.id != id) continue;
        if (exl3_expert_format_equal(entry.desc, desc)) return id;
        error = "EXL3 format hash collision";
        return 0;
    }
    entries_.push_back(Entry{id, desc});
    return id;
}

const Exl3ExpertFormatDesc* Exl3ExpertFormatRegistry::find(Exl3ExpertFormatID id) const {
    for (const Entry& entry : entries_)
        if (entry.id == id) return &entry.desc;
    return nullptr;
}

}  // namespace strata::core
