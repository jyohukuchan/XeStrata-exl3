// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Fused transform tails around EXL3 dots (exl3_fused.hpp).  One work-group owns one 128-wide
// block and runs the stages in sequence in its SG32, so the values pass through registers
// instead of through the buffers the separate kernels used; every stage computes exactly what
// the kernel it replaces computes - the same expressions, in the same order - and the launch
// count drops by the stages between the transforms.
#include "strata/kernels/exl3_fused.hpp"

#include "strata/core/runtime.hpp"
#include "device_caps.hpp"
#include "exl3_hadamard.hpp"

#include "strata/kernels/f16_bits.hpp"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strata::kernels::exl3 {
namespace {

using had::fwht128_input_registers;
using had::fwht128_output;

sycl::queue& queue_for(void* stream) { return core::Runtime::get().stream(stream); }

void finish(void* stream, const sycl::event& e, const char* what) {
    if (!stream) core::Runtime::get().wait(e, what);
}

bool block_shape_ok(const MatrixView& view, int64_t rows, int64_t width) {
    return rows > 0 && width > 0 && width % 128 == 0 && view.k > 0 && view.n > 0;
}

// D: double where the device has FP64 (the SwiGLU and the GDN norm's sums), float elsewhere
// (device_caps.hpp) - the same split the standalone kernels use.
template <typename D>
sycl::event submit_swiglu_tail(sycl::queue& queue, const MatrixView& gate, const MatrixView& up,
                               const MatrixView& down, float* gate_values, float* up_values, float* out,
                               int64_t rows, const std::vector<sycl::event>& deps) {
    const size_t width = static_cast<size_t>(gate.n);
    const size_t blocks = width / 128;
    return queue.submit([&](sycl::handler& h) {
        if (!deps.empty()) h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(rows) * blocks * 32),
                                         sycl::range<1>(32)),
                       [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
            const sycl::sub_group sg = item.get_sub_group();
            const int lane = static_cast<int>(sg.get_local_id()[0]);
            const size_t group = static_cast<size_t>(item.get_group_linear_id());
            const int64_t row = static_cast<int64_t>(group / blocks);
            const int64_t block = static_cast<int64_t>(group % blocks) * 128;
            const float* gd = gate_values + row * static_cast<int64_t>(gate.n);
            const float* ud = up_values + row * static_cast<int64_t>(up.n);
            float* wr = out + row * static_cast<int64_t>(down.k);
            float g0, g1, g2, g3, u0, u1, u2, u3;
            // The dots ran with their output stages off, so each value here is what the
            // standalone output transform produces: the transform when the view has `svh`,
            // then the bias either way (without `svh` the dot would have added it).
            if (gate.svh != nullptr)
                fwht128_output(sg, lane, gd, gate.svh, gate.bias, block, true, g0, g1, g2, g3);
            else {
                g0 = gd[block + lane]; g1 = gd[block + lane + 32];
                g2 = gd[block + lane + 64]; g3 = gd[block + lane + 96];
                if (gate.bias != nullptr) {
                    g0 += f32_from_f16(gate.bias[block + lane]);
                    g1 += f32_from_f16(gate.bias[block + lane + 32]);
                    g2 += f32_from_f16(gate.bias[block + lane + 64]);
                    g3 += f32_from_f16(gate.bias[block + lane + 96]);
                }
            }
            if (up.svh != nullptr)
                fwht128_output(sg, lane, ud, up.svh, up.bias, block, true, u0, u1, u2, u3);
            else {
                u0 = ud[block + lane]; u1 = ud[block + lane + 32];
                u2 = ud[block + lane + 64]; u3 = ud[block + lane + 96];
                if (up.bias != nullptr) {
                    u0 += f32_from_f16(up.bias[block + lane]);
                    u1 += f32_from_f16(up.bias[block + lane + 32]);
                    u2 += f32_from_f16(up.bias[block + lane + 64]);
                    u3 += f32_from_f16(up.bias[block + lane + 96]);
                }
            }
            // The SwiGLU of the standalone kernel: silu in D, then the gated multiply in FP32.
            g0 = (float) ((D) g0 / ((D) 1 + sycl::exp(-(D) g0))) * u0;
            g1 = (float) ((D) g1 / ((D) 1 + sycl::exp(-(D) g1))) * u1;
            g2 = (float) ((D) g2 / ((D) 1 + sycl::exp(-(D) g2))) * u2;
            g3 = (float) ((D) g3 / ((D) 1 + sycl::exp(-(D) g3))) * u3;
            if (down.suh != nullptr)
                fwht128_input_registers(sg, lane, down.suh, block, g0, g1, g2, g3);
            wr[block + lane] = g0; wr[block + lane + 32] = g1;
            wr[block + lane + 64] = g2; wr[block + lane + 96] = g3;
        });
    });
}

// The GDN output norm's warp sum, the same shuffle-down tree the standalone kernel adds in.
template <typename D>
inline D warp_sum(const sycl::sub_group& sg, D v) {
    for (int off = 16; off > 0; off >>= 1) v += sycl::permute_group_by_xor(sg, v, off);
    return v;
}

inline float sigmoid_f(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }

template <bool Silu, typename D>
sycl::event submit_gdn_tail(sycl::queue& queue, const MatrixView& gate, const MatrixView& out, const float* o,
                            float* z_values, const float* ssm_norm, float* y, int64_t rows, float eps,
                            const std::vector<sycl::event>& deps) {
    return queue.submit([&](sycl::handler& h) {
        if (!deps.empty()) h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(rows) * 32), sycl::range<1>(32)),
                       [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
            const sycl::sub_group sg = item.get_sub_group();
            const int lane = static_cast<int>(sg.get_local_id()[0]);
            const int64_t block = static_cast<int64_t>(item.get_group_linear_id()) * 128;
            const float* po = o;
            float* pz = z_values;
            float* py = y;
            const float* pn = ssm_norm;
            float a0, a1, a2, a3;
            if (gate.svh != nullptr)
                fwht128_output(sg, lane, pz, gate.svh, gate.bias, block, true, a0, a1, a2, a3);
            else {
                a0 = pz[block + lane]; a1 = pz[block + lane + 32];
                a2 = pz[block + lane + 64]; a3 = pz[block + lane + 96];
                if (gate.bias != nullptr) {
                    a0 += f32_from_f16(gate.bias[block + lane]);
                    a1 += f32_from_f16(gate.bias[block + lane + 32]);
                    a2 += f32_from_f16(gate.bias[block + lane + 64]);
                    a3 += f32_from_f16(gate.bias[block + lane + 96]);
                }
            }
            pz[block + lane] = a0; pz[block + lane + 32] = a1;
            pz[block + lane + 64] = a2; pz[block + lane + 96] = a3;
            // The output norm of the standalone kernel: the squared norm of the step's output in
            // D and the same warp sum, then the elementwise stage in the same order.
            const float o0 = po[block + lane], o1 = po[block + lane + 32];
            const float o2 = po[block + lane + 64], o3 = po[block + lane + 96];
            D acc = 0;
            acc += (D) o0 * (D) o0;
            acc += (D) o1 * (D) o1;
            acc += (D) o2 * (D) o2;
            acc += (D) o3 * (D) o3;
            const D ssum = sycl::group_broadcast(sg, warp_sum(sg, acc), 0);
            const float inv = (float) ((D) 1 / sycl::sqrt(ssum / (D) 128 + (D) eps));
            const float n0 = pn[lane], n1 = pn[lane + 32];
            const float n2 = pn[lane + 64], n3 = pn[lane + 96];
            const float g0 = sigmoid_f(a0), g1 = sigmoid_f(a1), g2 = sigmoid_f(a2), g3 = sigmoid_f(a3);
            a0 = o0 * inv * n0 * (Silu ? a0 * g0 : g0);
            a1 = o1 * inv * n1 * (Silu ? a1 * g1 : g1);
            a2 = o2 * inv * n2 * (Silu ? a2 * g2 : g2);
            a3 = o3 * inv * n3 * (Silu ? a3 * g3 : g3);
            if (out.suh != nullptr)
                fwht128_input_registers(sg, lane, out.suh, block, a0, a1, a2, a3);
            py[block + lane] = a0; py[block + lane + 32] = a1;
            py[block + lane + 64] = a2; py[block + lane + 96] = a3;
        });
    });
}

}  // namespace

bool exl3_fused_tail_supported(void* stream) {
    try {
        const auto subgroup_sizes =
            queue_for(stream).get_device().get_info<sycl::info::device::sub_group_sizes>();
        if (std::find(subgroup_sizes.begin(), subgroup_sizes.end(), size_t(32)) == subgroup_sizes.end()) return false;
        return queue_for(stream).get_device().get_info<sycl::info::device::max_work_group_size>() >= 32;
    } catch (...) {
        return false;
    }
}

bool exl3_fused_tail_enabled() {
    static const bool enabled = [] {
        const char* text = std::getenv("STRATA_EXL3_FUSED_TAILS");
        return text == nullptr || std::strtol(text, nullptr, 10) != 0;
    }();
    return enabled;
}

void exl3_fused_swiglu_tail_fp32(const MatrixView& gate, const MatrixView& up, const MatrixView& down,
                                 float* gate_values, float* up_values, float* out, int64_t rows, void* stream) {
    if (gate_values == nullptr || up_values == nullptr || out == nullptr ||
        !block_shape_ok(gate, rows, gate.n) || !block_shape_ok(up, rows, up.n) || down.k != gate.n ||
        gate.n != up.n)
        throw std::invalid_argument("exl3_fused_swiglu_tail_fp32: invalid spans");
    sycl::queue& queue = queue_for(stream);
    (void) (xe::has_fp64(queue)
                ? submit_swiglu_tail<double>(queue, gate, up, down, gate_values, up_values, out, rows, {})
                : submit_swiglu_tail<float>(queue, gate, up, down, gate_values, up_values, out, rows, {}));
}

void exl3_fused_gdn_tail_fp32(const MatrixView& gate, const MatrixView& out, const float* o, float* z_values,
                              const float* ssm_norm, float* y, int64_t rows, bool silu, float eps, void* stream) {
    if (o == nullptr || z_values == nullptr || ssm_norm == nullptr || y == nullptr ||
        !block_shape_ok(gate, rows, gate.n) || out.k != gate.n)
        throw std::invalid_argument("exl3_fused_gdn_tail_fp32: invalid spans");
    sycl::queue& queue = queue_for(stream);
    sycl::event done;
    if (xe::has_fp64(queue)) {
        if (silu)
            done = submit_gdn_tail<true, double>(queue, gate, out, o, z_values, ssm_norm, y, rows, eps, {});
        else
            done = submit_gdn_tail<false, double>(queue, gate, out, o, z_values, ssm_norm, y, rows, eps, {});
    } else {
        if (silu)
            done = submit_gdn_tail<true, float>(queue, gate, out, o, z_values, ssm_norm, y, rows, eps, {});
        else
            done = submit_gdn_tail<false, float>(queue, gate, out, o, z_values, ssm_norm, y, rows, eps, {});
    }
    // Like gdn_out_norm: a caller on the default stream wants the norm's result before it returns.
    finish(stream, done, "exl3_fused_gdn_tail_fp32");
}

}  // namespace strata::kernels::exl3
