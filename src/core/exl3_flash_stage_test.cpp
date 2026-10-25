// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// The EXL3 Flash layer split and the PLE's home stage.  CPU only: no model, no device.
//
// The split is what decides which stage runs the PLE layer, and the PLE's device half - its
// projections, its gathered row, its scratch and its conv history - belongs to that stage.  A
// split that left the PLE layer outside every stage, or that gave a stage no layer at all, would
// break the multi-GPU path in a way the first forward pass would show as a wrong answer rather
// than as an error, so the split is checked here for every shape it can take.

#include "strata/core/exl3_flash_model.hpp"
#include "strata/core/layer.hpp"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool split_is_sound(int64_t layers, int64_t stages) {
    int64_t covered = 0;
    int64_t previous_end = 0;
    for (int64_t index = 0; index < stages; ++index) {
        const strata::core::Exl3FlashStageRange range = strata::core::exl3_flash_stage_range(layers, stages, index);
        if (range.layer_begin != previous_end || range.layer_end <= range.layer_begin ||
            range.layer_end > layers) {
            std::cerr << "the split of " << layers << " layers over " << stages << " stages is not contiguous at stage "
                      << index << '\n';
            return false;
        }
        previous_end = range.layer_end;
        covered += range.layer_end - range.layer_begin;
    }
    if (covered != layers || previous_end != layers) {
        std::cerr << "the split of " << layers << " layers over " << stages << " stages covers " << covered << '\n';
        return false;
    }
    return true;
}

bool ple_home_holds_the_ple_layer(int64_t layers, int64_t stages) {
    const int64_t home = strata::core::exl3_flash_ple_stage(layers, stages);
    if (layers <= strata::core::PLE_LAYER) {
        // A model without the PLE layer has no home stage: open() refuses it, and the split must
        // not name a stage that would then run a layer the model does not have.
        if (home != -1) {
            std::cerr << "a model without the PLE layer named the home stage " << home << '\n';
            return false;
        }
        return true;
    }
    if (home < 0) {
        std::cerr << "no stage holds the PLE layer for " << layers << " layers over " << stages << " stages\n";
        return false;
    }
    const strata::core::Exl3FlashStageRange range = strata::core::exl3_flash_stage_range(layers, stages, home);
    if (range.layer_begin > strata::core::PLE_LAYER || strata::core::PLE_LAYER >= range.layer_end) {
        std::cerr << "the PLE's home stage " << home << " does not hold layer " << strata::core::PLE_LAYER << '\n';
        return false;
    }
    // No other stage may hold it: the PLE's device half is on the home stage alone.
    for (int64_t index = 0; index < stages; ++index) {
        if (index == home) continue;
        const strata::core::Exl3FlashStageRange other =
            strata::core::exl3_flash_stage_range(layers, stages, index);
        if (other.layer_begin <= strata::core::PLE_LAYER && strata::core::PLE_LAYER < other.layer_end) {
            std::cerr << "stage " << index << " holds the PLE layer as well\n";
            return false;
        }
    }
    return true;
}

// The explicit split (--stage-layers): the same soundness from the counts the caller gives, and
// refusals with a reason for the counts that are not a split at all.
bool explicit_split_is_sound(int64_t layers, const std::vector<int64_t>& counts) {
    std::vector<strata::core::Exl3FlashStageRange> ranges;
    std::string err;
    if (!strata::core::exl3_flash_stage_ranges(layers, static_cast<int64_t>(counts.size()), counts, ranges, err)) {
        std::cerr << "the explicit split of " << layers << " layers was refused: " << err << '\n';
        return false;
    }
    int64_t previous_end = 0;
    for (size_t index = 0; index < ranges.size(); ++index) {
        if (ranges[index].layer_begin != previous_end || ranges[index].layer_end <= ranges[index].layer_begin) {
            std::cerr << "the explicit split is not contiguous at stage " << index << '\n';
            return false;
        }
        previous_end = ranges[index].layer_end;
    }
    if (previous_end != layers) {
        std::cerr << "the explicit split covers " << previous_end << " of " << layers << " layers\n";
        return false;
    }
    if (layers > strata::core::PLE_LAYER) {
        const int64_t home = strata::core::exl3_flash_ple_stage(ranges);
        if (home < 0 || ranges[static_cast<size_t>(home)].layer_begin > strata::core::PLE_LAYER ||
            strata::core::PLE_LAYER >= ranges[static_cast<size_t>(home)].layer_end) {
            std::cerr << "the explicit split's home stage does not hold the PLE layer\n";
            return false;
        }
    }
    return true;
}

bool explicit_split_is_refused(int64_t layers, int64_t stages, const std::vector<int64_t>& counts, const char* what) {
    std::vector<strata::core::Exl3FlashStageRange> ranges;
    std::string err;
    if (strata::core::exl3_flash_stage_ranges(layers, stages, counts, ranges, err) || err.empty()) {
        std::cerr << "a split that is " << what << " was answered instead of refused\n";
        return false;
    }
    return true;
}

}  // namespace

int main() {
    const std::vector<int64_t> layer_counts = {1, 2, 3, 4, 8, 24, 48, 96};
    const std::vector<int64_t> stage_counts = {1, 2, 3, 4, 8, 16, 48};
    for (const int64_t layers : layer_counts) {
        for (const int64_t stages : stage_counts) {
            if (stages > layers) continue;
            if (!split_is_sound(layers, stages)) return 1;
            if (!ple_home_holds_the_ple_layer(layers, stages)) return 1;
        }
    }
    // The one stage keeps every layer, which is the split the single-stage path has always used.
    const strata::core::Exl3FlashStageRange single = strata::core::exl3_flash_stage_range(48, 1, 0);
    if (single.layer_begin != 0 || single.layer_end != 48) {
        std::cerr << "the single stage does not hold every layer\n";
        return 1;
    }
    // A shape that is not a split at all is refused instead of answering with an empty range.
    if (strata::core::exl3_flash_stage_range(48, 2, 2).layer_end != 0 ||
        strata::core::exl3_flash_ple_stage(0, 1) != -1) {
        std::cerr << "an invalid split was answered instead of refused\n";
        return 1;
    }
    // The explicit split (--stage-layers) gives the same coverage from the counts, uneven ones
    // included, and refuses the counts that are not a split of the model.
    if (!explicit_split_is_sound(48, {18, 18, 12}) || !explicit_split_is_sound(48, {47, 1}) ||
        !explicit_split_is_sound(48, {1, 47}) || !explicit_split_is_sound(48, {48}) ||
        !explicit_split_is_sound(1, {1}))
        return 1;
    if (!explicit_split_is_refused(48, 3, {18, 18, 13}, "more layers than the model") ||
        !explicit_split_is_refused(48, 3, {18, 18, 11}, "fewer layers than the model") ||
        !explicit_split_is_refused(48, 3, {24, 24, 0}, "a stage without a layer") ||
        !explicit_split_is_refused(48, 3, {24, -1, 25}, "a stage without a layer") ||
        !explicit_split_is_refused(48, 3, {24, 24}, "one count short of the stages"))
        return 1;
    // Batched prefill chunking: full chunks, one short last chunk, and no chunk at
    // all when the prompt runs token by token.
    using strata::core::exl3_flash_prefill_chunk_sizes;
    if (exl3_flash_prefill_chunk_sizes(2048, 256) != std::vector<int64_t>(8, 256) ||
        exl3_flash_prefill_chunk_sizes(2050, 256) !=
            std::vector<int64_t>({256, 256, 256, 256, 256, 256, 256, 256, 2}) ||
        exl3_flash_prefill_chunk_sizes(3, 256) != std::vector<int64_t>({3}) ||
        exl3_flash_prefill_chunk_sizes(1, 1) != std::vector<int64_t>({1}) ||
        !exl3_flash_prefill_chunk_sizes(0, 256).empty() ||
        !exl3_flash_prefill_chunk_sizes(10, 0).empty() ||
        !exl3_flash_prefill_chunk_sizes(10, -1).empty()) {
        std::cerr << "the prefill chunk sizes are wrong\n";
        return 1;
    }
    std::cout << "EXL3 Flash stage split checks: OK\n";
    return 0;
}
