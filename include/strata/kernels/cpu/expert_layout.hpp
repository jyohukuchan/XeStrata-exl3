// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// include/strata/kernels/cpu/expert_layout.hpp - plan v0.3 P6: where each routed expert lives in experts.bin.
//
// A Q2_0 pack (tools/strata_pack.py) has one blob size for every layer, `BLOB`, in the Strata expert form.  A
// pack that keeps its experts in their raw GGUF slices has a per-layer blob size; `native_experts.txt` said how,
// and is refused now that the engine no longer computes native (IQ) experts.  Everything that touches an
// expert blob - the arena, the VRAM tier, the prompt path, the CPU pool, the GPU window - asks this table.
#pragma once

#include "strata/kernels/cpu/expert.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::kernels::cpu {

/// One layer's expert geometry in its GGUF form (the record a native pack's native_experts.txt held).  The engines
/// size their per-layer buffers from it; the kernels that computed with such rows went with the native expert path.
struct NativeFmt {
    int gu_type = -1, d_type = -1;      ///< ggml types of gate/up and of down
    int gu_act = -1, d_act = -1;        ///< their vec_dot_type (the activation formats)
    /// The up rows' type: gu_type, or another one with the same activation format (Ornith-1.5's GSQ-RCO files mix
    /// Q2_K, Q3_K and Q4_K between gate and up in some layers).  gu_row and up_off are the gate rows'.
    int up_type = -1;
    int64_t n_embd = 0, n_ff = 0;
    size_t gu_row = 0, d_row = 0;       ///< bytes per weight row
    size_t up_row = 0;                  ///< bytes per up row (gu_row when up_type is gu_type)
    size_t up_off = 0, down_off = 0;    ///< inside the blob
    size_t bytes = 0;                   ///< the whole blob
    size_t act_bytes = 0, h_bytes = 0;  ///< quantized activation sizes (n_embd of gu_act, or the ActQ; n_ff of d_act)
};

/// Whether the layer's gate and up rows are of different types (one built another way leaves up_type at -1: the
/// gate's).
inline bool native_mixed_gu(const NativeFmt& f) { return f.up_type >= 0 && f.up_type != f.gu_type; }

struct ExpertLayout {
    bool native = false;
    int64_t n_layers = 0, n_expert = NE;
    std::vector<NativeFmt> fmt;           ///< per layer (a pack that kept its experts in GGUF form)
    std::vector<uint64_t> offset, bytes;  ///< per layer: where its 512 blobs start, bytes per blob
    /// Plan v0.3 P6: per layer, the absolute offsets of the gate / up / down tensors in their GGUF files, so the
    /// arena could be filled from the GGUF itself when the pack had no experts.bin (3 x n_layers, 0 = unknown).
    std::vector<uint64_t> gguf_off;
    /// Per layer, the GGUF files that held its gate / up / down tensors when the model's shards split the layers
    /// (3 x n_layers).  Empty = the model's main shard.
    std::vector<std::string> gguf_file;
    /// Per layer of this table, the model's block that holds it (`blk.<block>.ffn_*_exps`): a model whose first blocks
    /// have no routed experts (GLM-5.3's three dense ones) numbers its MoE layers apart from its blocks.  Empty: the
    /// same numbers.
    std::vector<int64_t> block;
    int version = 0;                      ///< native_experts.txt's header version (0 = none given)
    uint64_t max_blob = BLOB;
    uint64_t total = 0;                   ///< experts.bin size

    int64_t block_of(int64_t layer) const { return block.empty() ? layer : block[(size_t) layer]; }
    uint64_t blob_bytes(int64_t layer) const { return native ? bytes[(size_t) layer] : (uint64_t) BLOB; }
    uint64_t layer_offset(int64_t layer) const {
        return native ? offset[(size_t) layer] : (uint64_t) layer * (uint64_t) n_expert * (uint64_t) BLOB;
    }
    uint64_t blob_offset(int64_t layer, int64_t expert) const {
        return layer_offset(layer) + (uint64_t) expert * blob_bytes(layer);
    }
};

/// Plan v0.3 P6: whether this CPU (and its OS) runs the AVX-512 kernels (F, BW, VL, VNNI, VBMI).  Probed in a
/// file compiled without AVX-512, so asking is safe everywhere; STRATA_FORCE_AVX2=1 answers no (for tests).
bool cpu_avx512_ok();
/// Whether this CPU (and its OS) runs the AVX2 kernels (AVX, AVX2, FMA, F16C): the floor of every expert kernel
/// (q2_avx2.cpp and the EXL3 CPU kernels).  STRATA_FORCE_AVX2 does not change it.
bool cpu_avx2_ok();
/// Whether the AVX-VNNI kernels run here: the build has them (STRATA_HAVE_AVXVNNI), the CPU has AVX2 and AVX-VNNI
/// (CPUID 7.1 EAX bit 4).  STRATA_NO_AVXVNNI=1 answers no (to compare with the AVX2 kernels).
bool cpu_avxvnni_ok();
/// The CPU's brand string (CPUID 0x80000002..4), for messages; "unknown" when it has none.
std::string cpu_name();
/// Q2_0 GGUF rows / activation quantizer on the kernels this CPU has.
void q2_rows_any(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt, float* const* out,
                 int r0, int r1);
void act_quant_any(const float* x, int n, ActQ& a);

/// The process-wide layout (the canonical Q2_0 form).
const ExpertLayout& expert_layout();
/// Sets the canonical layout for `n_layers` x `n_expert` blobs; a pack that holds its experts in their raw GGUF
/// slices (native_experts.txt) is refused: the engine no longer computes native (IQ) experts.
bool expert_layout_load(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, std::string& err);
/// Replaces the process-wide layout.
void expert_layout_set(const ExpertLayout& layout);

}  // namespace strata::kernels::cpu
