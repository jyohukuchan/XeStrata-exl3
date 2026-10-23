// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/prefill/plain_prompt.cpp - PlainModel::prompt: a plain model's prompt in chunks of batched products.
//
// The block of plain_model.cpp, T tokens at a time: every projection a matrix product of the chunk's rows by the weight
// dequantized to FP16 (Gemm::native), the GDN conv and recurrence walking the chunk inside their kernels (the prompt
// path's, kernels.hpp), and the routed experts grouped by expert - each expert's weights read once for all the tokens
// routed to it, from its VRAM slot or copied from RAM, dequantized to FP16 and multiplied in one grouped product per
// group of experts.  The norms with the rotation, the gate, the K/V append, the attention over every cell, the router
// and the combine are the decode's kernels, called over the chunk's rows.
//
// The state is left as the windows leave it: the GDN recurrent state and conv history (the same layouts), the K/V
// cells of the chunk's positions.  The decode then continues at the next position.
#include "strata/core/plain_model.hpp"

#include "strata/core/gpu.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/iq_mmq.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/prompt_attn.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/rope_scaling.hpp"
#include "strata/kernels/router_top10.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/xmx_gemm.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <set>
#include <tuple>
#include <vector>

namespace strata::core {

namespace k = strata::kernels;

/// The products' state: Gemm over the dequantization scratch carved with the other buffers.
struct PlainModel::PromptGemm {
    prefill::Gemm gemm;
    prefill::mmq::Context mmq;
    bool identity = false;   // pb_.identity written
    // the int8 products' copies from RAM on a stream of their own, into the staging's two halves in turn: ready[h]
    // when half h holds its group, used[h] when that group's products are done with it
    strata::gpu::Stream xs = nullptr;
    strata::gpu::Event *ready[2] = {}, *used[2] = {};
    bool used_live[2] = {};
    ~PromptGemm() {
        if (xs != nullptr) strata::gpu::stream_sync(xs);
        for (int h = 0; h < 2; ++h) {
            if (ready[h] != nullptr) strata::gpu::event_destroy(ready[h]);
            if (used[h] != nullptr) strata::gpu::event_destroy(used[h]);
        }
        if (xs != nullptr) strata::gpu::stream_destroy(xs);
    }
    bool overlap() const { return xs != nullptr && ready[0] && ready[1] && used[0] && used[1]; }
};

void PlainModel::prompt_gemm_free(PromptGemm* g) { delete g; }

PlainModel::PromptGemm& PlainModel::prompt_gemm() {
    if (!prompt_gemm_) {
        prompt_gemm_ = std::unique_ptr<PromptGemm, void (*)(PromptGemm*)>(new PromptGemm, &PlainModel::prompt_gemm_free);
        prompt_gemm_->gemm.init_external(cs_, pb_.gemm_scratch, pb_.gemm_elems);
        PromptGemm& g = *prompt_gemm_;
        g.xs = strata::gpu::stream_create();
        for (int h = 0; h < 2; ++h) {
            if (!strata::gpu::event_create(&g.ready[h])) g.ready[h] = nullptr;
            if (!strata::gpu::event_create(&g.used[h])) g.used[h] = nullptr;
        }
        // without them the copies stay on the compute stream, in order with the products
    }
    return *prompt_gemm_;
}

bool PlainModel::prompt_borrow(int64_t T, std::string& err) {
    if (!settle_swaps(true, err)) return false;   // the tier's copies may still be writing the slots lent below
    const int64_t L = n_moe_, E = spec_.n_expert, n = n_slots_;
    const uint64_t* off = cache_.slot_offsets();
    if (borrow_first_ >= 0 || off == nullptr || L <= 0 || n < 2) { err = "plain prompt: no tier to borrow from"; return false; }
    // the chunk's device bytes from the tier's last slots: halved until they take at most half the tier, within its
    // last piece (the slots lie back to back only inside one)
    borrow_short_ = false;
    uint64_t db = 0, hb = 0;
    const int64_t want = T;
    int64_t first = n;
    const int64_t lowest = std::max<int64_t>(n / 2, cache_.last_piece_first());
    const int64_t least = std::clamp<int64_t>(want, 1, 16);   // a prompt under 16 tokens: one chunk of its length
    for (;; T /= 2) {
        if (T < least) break;
        prompt_carve(T, pb_, nullptr, nullptr, db, hb);
        first = n;
        while (first > lowest && off[n] - off[first] < db) --first;
        if (off[n] - off[first] >= db) break;
    }
    // the buffers of their own when their chunk is no shorter than the borrowed one: no slot lent, prompt_return
    // gives back none
    if (prompt_dev_ != nullptr && (T < least || std::min(want, prompt_dev_T_) >= T)) {
        const int64_t pieces = (want + prompt_dev_T_ - 1) / prompt_dev_T_;   // even chunks: no short last one
        T = (want + pieces - 1) / pieces;
        borrowed_res_.clear();
        borrow_first_ = n;
        prompt_carve(T, pb_, (uint8_t*) prompt_dev_, (uint8_t*) prompt_host_, db, hb);
        PromptGemm& g = prompt_gemm();
        g.gemm.rebind(pb_.gemm_scratch, pb_.gemm_elems);
        g.identity = false;
        return true;
    }
    if (T < least) {
        err = "plain prompt: the VRAM tier is too small for the prompt's buffers (" + std::to_string(least) +
              " tokens need " + std::to_string(db >> 20) + " MiB, the tier lends at most " +
              std::to_string((off[n] - off[lowest]) >> 20) + " MiB of its " + std::to_string(off[n] >> 20) + ")";
        borrow_short_ = true;
        return false;
    }
    borrowed_res_.clear();
    for (int64_t l = 0; l < L; ++l)
        for (int64_t e = 0; e < E; ++e) {
            int32_t& r = res_[(size_t) (l * E + e)];
            if (r >= first) {
                borrowed_res_.emplace_back((int32_t) l, (int32_t) e);
                r = -1;
            }
        }
    for (auto& f : free_) f.erase(std::remove_if(f.begin(), f.end(), [&](int32_t x) { return x >= first; }), f.end());
    borrow_first_ = first;
    prompt_carve(T, pb_, cache_.device_slot((int32_t) first), (uint8_t*) prompt_host_, db, hb);
    PromptGemm& g = prompt_gemm();
    g.gemm.rebind(pb_.gemm_scratch, pb_.gemm_elems);
    g.identity = false;   // the identity map is rewritten where the buffers now lie
    return true;
}

bool PlainModel::prompt_return(std::string& err) {
    const int64_t L = n_moe_, n = n_slots_;
    if (borrow_first_ < 0 || L <= 0) return true;   // a borrow needs slots, and slots need MoE layers
    const auto& lay = k::cpu::expert_layout();
    if (!strata::gpu::stream_sync(cs_)) { err = std::string("plain prompt: ") + strata::gpu::last_error(); return false; }
    for (int64_t s = borrow_first_; s < n; ++s) free_[(size_t) (s % L)].push_back((int32_t) s);
    borrow_first_ = -1;
    // the experts the prompt displaced, back into the slots: straight from pinned RAM, else through fill_slot
    for (const auto& [l, e] : borrowed_res_) {
        auto& f = free_[(size_t) l];
        if (f.empty()) break;
        const uint8_t* hb = src_->transient(l, e) ? nullptr : src_->blob(l, e);
        if (hb != nullptr && strata::gpu::is_host_usm(hb)) {
            const int32_t slot = f.back();
            if (!strata::gpu::copy_async(cache_.device_slot(slot), hb, lay.blob_bytes(l), cs_)) {
                err = std::string("plain prompt: ") + strata::gpu::last_error();
                return false;
            }
            f.pop_back();
            res_[(size_t) (l * spec_.n_expert + e)] = slot;
        } else if (!fill_slot(l, e, err)) {
            return false;
        }
    }
    borrowed_res_.clear();
    if (!strata::gpu::stream_sync(cs_)) { err = std::string("plain prompt: ") + strata::gpu::last_error(); return false; }
    return true;
}

bool PlainModel::prompt_warm(std::string& err) {
    // EXL3 dense projections and experts use their runtime owners.  The
    // legacy batched-product warmup below assumes GGUF W.data and an
    // ExpertCache, neither of which exists in EXL3 mode.
    if (exl3_mode_) return true;
    if (opt_.prompt_chunk <= 0 || prompt_host_ == nullptr) return true;
    if (!prompt_borrow(std::min<int64_t>(opt_.prompt_chunk, 512), err)) return false;
    const bool ok = prompt_warm_products(err);
    std::string err2;
    if (!prompt_return(err2) && ok) { err = err2; return false; }
    return ok;
}

bool PlainModel::prompt_warm_products(std::string& err) {
    // A library's first product of a kind loads its kernels for it, inside whichever prompt comes first (rocBLAS on
    // the RX 9060 XT: 0.5 s for the first product, 0.9 s more for the first of 32 rows): each dense shape the prompt
    // takes, once, on a chunk of zeros into the buffers the prompt overwrites.
    PromptBufs& P = pb_;
    prefill::Gemm& gm = prompt_gemm().gemm;
    const int64_t T = P.T, HV = spec_.ssm_dt_rank;
    std::set<std::tuple<int, int, int, int64_t>> seen;
    auto warm = [&](const W& w, float* y, int64_t ldy = 0) {
        if (w.data != nullptr && w.type != 0 && seen.insert({w.type, w.n_out, w.n_in, ldy}).second)
            gm.native(P.x16, w.type, w.data, y, T, w.n_out, w.n_in, ldy);
    };
    try {
        if (!strata::gpu::memset_async(P.x16, 0, (size_t) (T * spec_.n_embd) * 2, cs_)) {
            err = std::string("plain prompt: warming: ") + strata::gpu::last_error();
            return false;
        }
        std::vector<const Block*> all;
        for (const Block& b : blocks_) all.push_back(&b);
        if (has_mtp_) {
            all.push_back(&mtp_);
            warm(mtp_eh_, P.x);
        }
        for (const Block* bp : all) {
            const Block& b = *bp;
            if (b.recurrent) {
                warm(b.qkv, P.qkv);
                warm(b.z, P.z);
                warm(b.alpha, P.ab, 2 * HV);
                warm(b.beta, P.ab + HV, 2 * HV);
                warm(b.ssm_out, P.mix);
            } else {
                warm(b.q, P.qfull);
                warm(b.k, P.kraw);
                warm(b.v, P.vcur);
                warm(b.o, P.mix);
            }
            if (has_shared_) {
                warm(b.sh_gate, P.sh_gate);
                warm(b.sh_up, P.sh_up);
                warm(b.sh_down, P.shared);
            }
        }
    } catch (const std::exception& e) {
        err = std::string("plain prompt: warming: ") + e.what();
        return false;
    }
    if (!strata::gpu::stream_sync(cs_)) {
        err = std::string("plain prompt: warming: ") + strata::gpu::last_error();
        return false;
    }
    return true;
}

bool PlainModel::prompt(const int32_t* tokens, int64_t n, int64_t pos, int32_t* out, float* logits, std::string& err) {
    if (exl3_mode_) {
        if (opt_.prompt_chunk <= 0 || n <= 0) { err.clear(); return false; }
        if (tokens == nullptr || out == nullptr) {
            err = "plain prompt: null token or output buffer";
            return false;
        }
        if (pos < 0 || n > opt_.max_cells - pos) {
            err = "plain prompt: past the context";
            return false;
        }
        const int64_t chunk = std::max<int64_t>(1, std::min<int64_t>(opt_.prompt_chunk, opt_.max_t));
        std::vector<int32_t> chunk_out((size_t) chunk);
        std::vector<float> chunk_logits;
        if (logits != nullptr) chunk_logits.resize((size_t) chunk * (size_t) n_vocab_);
        for (int64_t offset = 0; offset < n;) {
            const int T = (int) std::min<int64_t>(chunk, n - offset);
            const bool last = offset + T == n;
            if (!forward(tokens + offset, T, pos + offset, chunk_out.data(),
                         last && logits != nullptr ? chunk_logits.data() : nullptr, err))
                return false;
            if (last) {
                *out = chunk_out[(size_t) T - 1];
                if (logits != nullptr)
                    std::memcpy(logits, chunk_logits.data() + (size_t) (T - 1) * (size_t) n_vocab_,
                                (size_t) n_vocab_ * sizeof(float));
            }
            offset += T;
        }
        return true;
    }
    // The batched prompt over a GGUF model's embedding went with the GGUF model path: a prompt now runs through
    // the decode windows, which is what the EXL3 mode already did.
    err = "plain prompt: this engine runs EXL3 archives; a prompt runs through the decode windows";
    return false;
}

}  // namespace strata::core
