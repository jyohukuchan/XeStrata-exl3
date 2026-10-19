// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "strata/core/exl3_expert_formats.hpp"

#include <cstdio>
#include <cstdint>
#include <limits>
#include <string>

namespace {

strata::core::Exl3ExpertFormatDesc valid_format() {
    strata::core::Exl3ExpertFormatDesc d;
    d.alignment = 2;
    d.gate.k = d.up.k = 128; d.gate.n = d.up.n = 256;
    d.down.k = 256; d.down.n = 128;
    d.gate.bits = 2; d.gate.codebook = 0;
    d.up.bits = 3; d.up.codebook = 1;
    d.down.bits = 5; d.down.codebook = 2;
    uint64_t at = 0;
    auto fill = [&](auto& p, uint64_t trellis) {
        p.trellis = {at, trellis}; at += trellis;
        p.suh = {at, (uint64_t) p.k * 2}; at += p.suh.bytes;
        p.svh = {at, (uint64_t) p.n * 2}; at += p.svh.bytes;
        p.bias = {at, (uint64_t) p.n * 2}; at += p.bias.bytes;
    };
    fill(d.gate, 128 * 256 * 2 / 8);
    fill(d.up, 128 * 256 * 3 / 8);
    fill(d.down, 256 * 128 * 5 / 8);
    d.total_bytes = at;  // the artifact factory stores the exact cursor, aligned component starts only
    return d;
}

bool check(bool condition, const char* text) {
    if (!condition) std::fprintf(stderr, "FAIL: %s\n", text);
    return condition;
}

}  // namespace

int main() {
    using namespace strata::core;
    int failures = 0;
    Exl3ExpertFormatDesc a = valid_format(), b = a;
    std::string error;
    if (!check(validate_exl3_expert_format(a, error), "valid descriptor accepted")) ++failures;
    if (!check(exl3_expert_format_equal(a, b), "logical descriptor equality")) ++failures;
    const Exl3ExpertFormatID id_a = exl3_expert_format_id(a);
    if (!check(id_a != 0, "valid descriptor gets nonzero id")) ++failures;
    b.gate.bias.bytes = 0;
    if (!check(!exl3_expert_format_equal(a, b) && exl3_expert_format_id(a) != exl3_expert_format_id(b),
               "logical field changes alter id"))
        ++failures;
    Exl3ExpertFormatDesc scale_offset = a;
    scale_offset.gate.suh.offset += 2;
    scale_offset.gate.svh.offset += 2;
    scale_offset.gate.bias.offset += 2;
    scale_offset.up.trellis.offset += 2;
    scale_offset.up.suh.offset += 2;
    scale_offset.up.svh.offset += 2;
    scale_offset.up.bias.offset += 2;
    scale_offset.down.trellis.offset += 2;
    scale_offset.down.suh.offset += 2;
    scale_offset.down.svh.offset += 2;
    scale_offset.down.bias.offset += 2;
    scale_offset.total_bytes += 2;
    if (!check(validate_exl3_expert_format(scale_offset, error) &&
                   exl3_expert_format_id(scale_offset) != exl3_expert_format_id(a),
               "scale offset changes alter id"))
        ++failures;

    Exl3ExpertFormatRegistry registry;
    const Exl3ExpertFormatID registered_a = registry.register_format(a, error);
    const Exl3ExpertFormatID registered_again = registry.register_format(a, error);
    if (!check(registered_a == id_a && registered_again == id_a && registry.size() == 1, "registry deduplicates exact format"))
        ++failures;
    if (!check(registry.find(id_a) != nullptr && exl3_expert_format_equal(*registry.find(id_a), a), "registry lookup"))
        ++failures;

    Exl3ExpertFormatDesc overlap = a;
    overlap.up.trellis.offset = overlap.gate.trellis.offset;
    if (!check(!validate_exl3_expert_format(overlap, error), "overlapping components rejected")) ++failures;
    Exl3ExpertFormatDesc bad_bits = a;
    bad_bits.gate.bits = 0;
    if (!check(exl3_expert_format_id(bad_bits) == 0, "invalid bits rejected")) ++failures;

    Exl3ExpertFormatDesc overflow = a;
    const int64_t huge = std::numeric_limits<int64_t>::max() - (std::numeric_limits<int64_t>::max() % 128);
    overflow.gate.k = overflow.up.k = huge;
    overflow.gate.n = overflow.up.n = 128;
    overflow.down.k = 128;
    overflow.down.n = huge;
    if (!check(!validate_exl3_expert_format(overflow, error), "wrapped trellis geometry rejected")) ++failures;

    std::printf("exl3_expert_formats_test: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
