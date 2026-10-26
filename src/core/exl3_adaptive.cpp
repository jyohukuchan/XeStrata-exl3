// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// The EXL3 adaptive tier's swap choice (see exl3_adaptive.hpp).
#include "strata/core/exl3_adaptive.hpp"

#include <algorithm>
#include <cstddef>
#include <utility>

namespace strata::core {

std::vector<Exl3AdaptSwap> plan_adaptive_swaps(int64_t n_layers, int64_t n_expert, int64_t layer_begin,
                                               int64_t layer_end, const double* usage, const uint8_t* resident,
                                               int64_t cap, double min_count, double min_gain) {
    std::vector<Exl3AdaptSwap> swaps;
    if (n_layers <= 0 || n_expert <= 0 || usage == nullptr || resident == nullptr) return swaps;
    layer_begin = std::max<int64_t>(layer_begin, 0);
    layer_end = std::min<int64_t>(layer_end, n_layers);
    if (layer_begin >= layer_end) return swaps;
    std::vector<std::pair<double, int32_t>> cand, vict;
    for (int64_t l = layer_begin; l < layer_end; ++l) {
        cand.clear();
        vict.clear();
        const double* u = usage + (size_t) l * (size_t) n_expert;
        const uint8_t* r = resident + (size_t) l * (size_t) n_expert;
        for (int32_t e = 0; e < (int32_t) n_expert; ++e) {
            if (r[e] != 0) vict.emplace_back(u[e], e);
            else if (u[e] > 0.0) cand.emplace_back(u[e], e);
        }
        if (cand.empty() || vict.empty()) continue;
        std::sort(cand.begin(), cand.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        size_t keep = 0;
        while (keep < cand.size() && cand[keep].first >= min_count) ++keep;
        cand.resize(keep);
        if (cand.empty()) continue;
        const size_t nc = std::min(cand.size(), vict.size());
        std::partial_sort(vict.begin(), vict.begin() + (ptrdiff_t) nc, vict.end(),
                          [](const auto& a, const auto& b) { return a.first < b.first; });
        for (size_t i = 0; i < nc; ++i) {
            if (cand[i].first < vict[i].first + min_gain) break;
            swaps.push_back({cand[i].first - vict[i].first, (int32_t) l, cand[i].second, vict[i].second});
        }
    }
    std::sort(swaps.begin(), swaps.end(), [](const Exl3AdaptSwap& a, const Exl3AdaptSwap& b) {
        if (a.gain != b.gain) return a.gain > b.gain;
        if (a.layer != b.layer) return a.layer < b.layer;
        return a.in < b.in;
    });
    if (cap > 0 && (int64_t) swaps.size() > cap) swaps.resize((size_t) cap);
    return swaps;
}

}  // namespace strata::core
