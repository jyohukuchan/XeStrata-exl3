// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// The Qwen3.8 MTP role gate: a draft block with a missing projection must fail
// closed before any weight is bound, so the gate is checked here against
// synthetic maps for every role it guards.

#include "strata/core/exl3_flash_model.hpp"

#include "strata/artifact/exl3_model_map.hpp"

#include <cstdint>
#include <iostream>
#include <string>

namespace {

using strata::artifact::exl3::ModelMap;
using strata::artifact::exl3::ProjectionKind;
using strata::artifact::exl3::ProjectionMapping;

void add_role(ModelMap& map, ProjectionKind kind, int count) {
    for (int i = 0; i < count; ++i) {
        ProjectionMapping item;
        item.kind = kind;
        map.projections.push_back(item);
    }
}

ModelMap complete_map(int64_t experts) {
    ModelMap map;
    map.geometry.layers = 1;
    map.geometry.experts = experts;
    add_role(map, ProjectionKind::mtp_fc, 2);
    add_role(map, ProjectionKind::qsa_q, 1);
    add_role(map, ProjectionKind::qsa_k, 1);
    add_role(map, ProjectionKind::qsa_v, 1);
    add_role(map, ProjectionKind::qsa_out, 1);
    add_role(map, ProjectionKind::qsa_indexer_q, 1);
    add_role(map, ProjectionKind::qsa_indexer_k, 1);
    add_role(map, ProjectionKind::hc_norm, 2);
    add_role(map, ProjectionKind::hc_down, 2);
    add_role(map, ProjectionKind::hc_up, 2);
    add_role(map, ProjectionKind::hc_inject, 2);
    add_role(map, ProjectionKind::expert_gate, static_cast<int>(experts));
    add_role(map, ProjectionKind::expert_up, static_cast<int>(experts));
    add_role(map, ProjectionKind::expert_down, static_cast<int>(experts));
    return map;
}

int failures = 0;

void check(bool condition, const char* name) {
    std::cout << (condition ? "OK   " : "FAIL ") << name << '\n';
    if (!condition) ++failures;
}

}  // namespace

int main() {
    {
        ModelMap map = complete_map(4);
        std::string err = "stale";
        check(strata::core::exl3_flash_mtp_roles_complete(map, err) && err.empty(), "complete map accepted");
    }
    {
        ModelMap map;
        std::string err;
        check(!strata::core::exl3_flash_mtp_roles_complete(map, err) && !err.empty(), "empty map rejected");
    }
    {
        ModelMap map = complete_map(4);
        map.geometry.layers = 0;
        std::string err;
        check(!strata::core::exl3_flash_mtp_roles_complete(map, err) && !err.empty(), "zero layers rejected");
    }
    {
        ModelMap map = complete_map(0);
        std::string err;
        check(!strata::core::exl3_flash_mtp_roles_complete(map, err) && !err.empty(), "zero experts rejected");
    }
    {
        ModelMap map = complete_map(4);
        map.projections.erase(map.projections.begin());
        std::string err;
        check(!strata::core::exl3_flash_mtp_roles_complete(map, err) && !err.empty(), "missing fc rejected");
    }
    {
        ModelMap map = complete_map(4);
        for (auto it = map.projections.begin(); it != map.projections.end(); ++it) {
            if (it->kind == ProjectionKind::qsa_v) {
                map.projections.erase(it);
                break;
            }
        }
        std::string err;
        check(!strata::core::exl3_flash_mtp_roles_complete(map, err) && !err.empty(), "missing qsa rejected");
    }
    {
        ModelMap map = complete_map(4);
        for (auto it = map.projections.begin(); it != map.projections.end(); ++it) {
            if (it->kind == ProjectionKind::hc_up) {
                map.projections.erase(it);
                break;
            }
        }
        std::string err;
        check(!strata::core::exl3_flash_mtp_roles_complete(map, err) && !err.empty(), "missing hc rejected");
    }
    {
        ModelMap map = complete_map(4);
        for (auto it = map.projections.begin(); it != map.projections.end(); ++it) {
            if (it->kind == ProjectionKind::expert_down) {
                map.projections.erase(it);
                break;
            }
        }
        std::string err;
        check(!strata::core::exl3_flash_mtp_roles_complete(map, err) && !err.empty(), "missing expert rejected");
    }
    if (failures != 0) {
        std::cerr << failures << " MTP role check(s) failed\n";
        return 1;
    }
    std::cout << "all MTP role checks passed\n";
    return 0;
}
