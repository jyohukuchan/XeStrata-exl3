// SPDX-FileCopyrightText: 2025 Turboderp
// SPDX-FileCopyrightText: 2026 jyohukuchan
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// EXL3 codebook rules follow exllamav3's MIT-licensed implementation.
// src/kernels/exl3_cb2_parity.cpp - the codebook 2 decode's two forms, checked exhaustively.
//
// The half fused multiply add of the reference decoder and the portable form return the same F16 bits
// for every byte sum; this program checks that on the device, for every byte sum (0..1020) and for
// every 16-bit state (0..65535), and checks that the projections themselves return the same values
// with the half form and with the portable form.  It checks the two byte-sum forms as well: the
// unsigned dot product and the signed one with the sign bits flipped must return the same sum, and
// the host's reference must agree with both, for every byte sum and for every state.
#include "strata/kernels/exl3.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "xe/exl3_cb2.hpp"

#include "parity_device.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

using strata::kernels::exl3::MatrixView;

constexpr uint32_t kMul1 = 0x83DCD12Du;
constexpr int kMaxByteSum = 1020;
// gitleaks:allow -- public EXL3 codebook multipliers and bit masks, never credentials.

// The codebook 2 value of a byte sum, from its definition: (887 * B - 452608) * 2^-17, rounded once to
// F16 with round to nearest even.  The exact value is a multiple of 2^-17 below 2^19, so it is exact in
// F32 and the conversion rounds it once.
uint16_t oracle_bits(int byte_sum) {
    const double exact = (double) (887 * (int64_t) byte_sum - 452608) * 0x1.0p-17;
    const float narrowed = (float) exact;
    if ((double) narrowed != exact) {
        std::fprintf(stderr, "exl3_cb2_parity: the oracle value of byte sum %d is not exact in F32\n", byte_sum);
        std::exit(1);
    }
    return strata::kernels::f16_from_f32(narrowed);
}

// The byte sum of a product, on the host: the four bytes of the product.
int host_byte_sum_of_product(uint32_t product) {
    return (int) ((product & 0xFFu) + ((product >> 8) & 0xFFu) + ((product >> 16) & 0xFFu) +
                  ((product >> 24) & 0xFFu));
}

// The byte sum of a state, on the host: the four bytes of state * 0x83DCD12D.
int host_byte_sum(uint16_t state) {
    return host_byte_sum_of_product((uint32_t) state * kMul1);
}

uint32_t float_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

// Every byte sum, decoded on the device in both forms.
bool check_byte_sums(sycl::queue& queue, strata::parity::Device& device, bool half_available) {
    constexpr size_t count = (size_t) kMaxByteSum + 1;
    uint16_t* d_portable_bits = device.alloc<uint16_t>(count);
    uint16_t* d_half_bits = device.alloc<uint16_t>(count);
    float* d_portable_values = device.alloc<float>(count);
    float* d_half_values = device.alloc<float>(count);
    queue.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(count), [=](sycl::id<1> id) {
            const int byte_sum = (int) id[0];
            d_portable_bits[id[0]] = strata::kernels::exl3::cb2_portable_bits(byte_sum);
            d_portable_values[id[0]] = strata::kernels::exl3::cb2_portable_f32(byte_sum);
            if (half_available) {
                d_half_bits[id[0]] = strata::kernels::exl3::cb2_half_value_bits(byte_sum);
                d_half_values[id[0]] = strata::kernels::exl3::cb2_half_f32(byte_sum);
            }
        });
    }).wait_and_throw();
    std::vector<uint16_t> portable_bits(count), half_bits(count);
    std::vector<float> portable_values(count), half_values(count);
    strata::parity::copy_out(portable_bits.data(), d_portable_bits, count);
    strata::parity::copy_out(portable_values.data(), d_portable_values, count);
    strata::parity::copy_out(half_bits.data(), d_half_bits, count);
    strata::parity::copy_out(half_values.data(), d_half_values, count);

    size_t bad = 0;
    for (size_t i = 0; i < count; ++i) {
        const uint16_t want = oracle_bits((int) i);
        bool ok = portable_bits[i] == want;
        if (half_available) {
            ok = ok && half_bits[i] == want;
            ok = ok && float_bits(half_values[i]) == float_bits(portable_values[i]);
        }
        if (!ok) {
            if (bad < 8)
                std::printf("byte sum %3zu: portable %04x half %04x oracle %04x\n", i, portable_bits[i],
                            half_available ? half_bits[i] : 0, want);
            ++bad;
        }
    }
    std::printf("byte sums 0..%d: %s (%zu of %zu bad, %s form)\n", kMaxByteSum, bad == 0 ? "OK" : "FAIL", bad, count,
                half_available ? "half" : "portable only");
    return bad == 0;
}

// The two byte-sum forms, for every byte sum and for every 16-bit state: the unsigned dot product and the
// signed one with the sign bits flipped must return the same sum, and the host's reference must agree with
// both.  The F16 operand of the half form (0x6400 + B) must be the same number as well.
bool check_byte_sum_forms(sycl::queue& queue, strata::parity::Device& device) {
    constexpr size_t sum_count = (size_t) kMaxByteSum + 1;
    constexpr size_t state_count = 65536;
    constexpr size_t count = sum_count + state_count;
    int* d_unsigned = device.alloc<int>(count);
    int* d_signed = device.alloc<int>(count);
    uint16_t* d_operands = device.alloc<uint16_t>(count);
    int* d_native = device.alloc<int>(1);
    queue.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(count), [=](sycl::id<1> id) {
            const size_t i = id[0];
            // In the first part the product is the byte sum itself (its other three bytes are zero); in the
            // second it is the mul1 product of a state.
            const uint32_t product = i < sum_count ? (uint32_t) i : (uint32_t) (i - sum_count) * kMul1;
            const int unsigned_sum = strata::kernels::exl3::cb2_byte_sum(product);
            const int signed_sum = strata::kernels::exl3::cb2_byte_sum_signed(product);
            d_unsigned[i] = unsigned_sum;
            d_signed[i] = signed_sum;
            d_operands[i] = strata::kernels::exl3::cb2_half_bits(unsigned_sum);
        });
    }).wait_and_throw();
    queue.submit([&](sycl::handler& h) {
        h.single_task([=] { d_native[0] = strata::kernels::exl3::cb2_unsigned_dot_native() ? 1 : 0; });
    }).wait_and_throw();
    std::vector<int> unsigned_sums(count), signed_sums(count), native(1, 0);
    std::vector<uint16_t> operands(count);
    strata::parity::copy_out(unsigned_sums.data(), d_unsigned, count);
    strata::parity::copy_out(signed_sums.data(), d_signed, count);
    strata::parity::copy_out(operands.data(), d_operands, count);
    strata::parity::copy_out(native.data(), d_native, 1);

    size_t bad = 0;
    for (size_t i = 0; i < count; ++i) {
        const uint32_t product = i < sum_count ? (uint32_t) i : (uint32_t) (i - sum_count) * kMul1;
        const int want = host_byte_sum_of_product(product);
        const bool ok = unsigned_sums[i] == want && signed_sums[i] == want &&
                        operands[i] == (uint16_t) (0x6400u + (unsigned) want);
        if (!ok) {
            if (bad < 8)
                std::printf("product %08x: unsigned %d signed %d host %d operand %04x\n", product, unsigned_sums[i],
                            signed_sums[i], want, operands[i]);
            ++bad;
        }
    }
    std::printf("byte-sum forms (%zu byte sums, %zu states): %s (%zu of %zu bad, %s dot product)\n", sum_count,
                state_count, bad == 0 ? "OK" : "FAIL", bad, count, native[0] ? "unsigned" : "signed");
    return bad == 0;
}

// Every 16-bit state, decoded on the device in both forms.  The state is what the grouped path extracts
// from a compressed tile, and its byte sum is what the decode turns into a value.
bool check_states(sycl::queue& queue, strata::parity::Device& device, bool half_available) {
    constexpr size_t count = 65536;
    int* d_sums = device.alloc<int>(count);
    int* d_signed_sums = device.alloc<int>(count);
    uint16_t* d_portable_bits = device.alloc<uint16_t>(count);
    uint16_t* d_half_bits = device.alloc<uint16_t>(count);
    queue.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(count), [=](sycl::id<1> id) {
            const uint16_t state = (uint16_t) id[0];
            const uint32_t product = (uint32_t) state * kMul1;
            const int byte_sum = strata::kernels::exl3::cb2_byte_sum(product);
            d_sums[id[0]] = byte_sum;
            d_signed_sums[id[0]] = strata::kernels::exl3::cb2_byte_sum_signed(product);
            d_portable_bits[id[0]] = strata::kernels::exl3::cb2_portable_bits(byte_sum);
            if (half_available) d_half_bits[id[0]] = strata::kernels::exl3::cb2_half_value_bits(byte_sum);
        });
    }).wait_and_throw();
    std::vector<int> sums(count), signed_sums(count);
    std::vector<uint16_t> portable_bits(count), half_bits(count);
    strata::parity::copy_out(sums.data(), d_sums, count);
    strata::parity::copy_out(signed_sums.data(), d_signed_sums, count);
    strata::parity::copy_out(portable_bits.data(), d_portable_bits, count);
    strata::parity::copy_out(half_bits.data(), d_half_bits, count);

    size_t bad = 0;
    bool seen[1021] = {};
    size_t distinct = 0;
    for (size_t i = 0; i < count; ++i) {
        const int want_sum = host_byte_sum((uint16_t) i);
        const uint16_t want_bits = oracle_bits(want_sum);
        bool ok = sums[i] == want_sum && signed_sums[i] == want_sum && portable_bits[i] == want_bits;
        if (half_available) ok = ok && half_bits[i] == want_bits;
        if (!ok) {
            if (bad < 8)
                std::printf("state %5zu: sum %d signed %d (want %d) portable %04x half %04x oracle %04x\n", i, sums[i],
                            signed_sums[i], want_sum, portable_bits[i], half_available ? half_bits[i] : 0, want_bits);
            ++bad;
        }
        if (!seen[want_sum]) {
            seen[want_sum] = true;
            ++distinct;
        }
    }
    std::printf("states 0..65535: %s (%zu of %zu bad, %zu distinct byte sums, %s form)\n", bad == 0 ? "OK" : "FAIL",
                bad, count, distinct, half_available ? "half" : "portable only");
    return bad == 0;
}

// The product form of the decode, for every 16-bit state: the half form takes the F16 operand's bias
// from the byte dot product's accumulator, the portable form needs the plain byte sum, and both must
// return the bits the definition gives.
bool check_product_forms(sycl::queue& queue, strata::parity::Device& device) {
    constexpr size_t count = 65536;
    uint16_t* d_half_bits = device.alloc<uint16_t>(count);
    uint16_t* d_portable_bits = device.alloc<uint16_t>(count);
    float* d_half_values = device.alloc<float>(count);
    float* d_portable_values = device.alloc<float>(count);
    queue.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(count), [=](sycl::id<1> id) {
            const uint32_t product = (uint32_t) (uint16_t) id[0] * kMul1;
            d_half_bits[id[0]] = strata::kernels::exl3::cb2_product_value_bits<true>(product);
            d_portable_bits[id[0]] = strata::kernels::exl3::cb2_product_value_bits<false>(product);
            d_half_values[id[0]] = strata::kernels::exl3::cb2_product_value_f32<true>(product);
            d_portable_values[id[0]] = strata::kernels::exl3::cb2_product_value_f32<false>(product);
        });
    }).wait_and_throw();
    std::vector<uint16_t> half_bits(count), portable_bits(count);
    std::vector<float> half_values(count), portable_values(count);
    strata::parity::copy_out(half_bits.data(), d_half_bits, count);
    strata::parity::copy_out(portable_bits.data(), d_portable_bits, count);
    strata::parity::copy_out(half_values.data(), d_half_values, count);
    strata::parity::copy_out(portable_values.data(), d_portable_values, count);

    size_t bad = 0;
    for (size_t i = 0; i < count; ++i) {
        const uint16_t want = oracle_bits(host_byte_sum((uint16_t) i));
        const bool ok = half_bits[i] == want && portable_bits[i] == want &&
                        float_bits(half_values[i]) == float_bits(portable_values[i]);
        if (!ok) {
            if (bad < 8)
                std::printf("state %5zu: half %04x portable %04x oracle %04x\n", i, half_bits[i], portable_bits[i],
                            want);
            ++bad;
        }
    }
    std::printf("product forms (65536 states): %s (%zu of %zu bad)\n", bad == 0 ? "OK" : "FAIL", bad, count);
    return bad == 0;
}

struct FixtureCase {
    uint32_t bits = 0;
    uint32_t codebook = 0;
    uint32_t k = 0;
    uint32_t n = 0;
    std::vector<uint16_t> trellis;
    std::vector<uint16_t> suh;
    std::vector<uint16_t> svh;
    std::vector<uint16_t> bias;
    std::vector<float> x;
};

bool read_bytes(const std::vector<uint8_t>& bytes, size_t& at, void* dst, size_t count) {
    if (count > bytes.size() - at) return false;
    std::memcpy(dst, bytes.data() + at, count);
    at += count;
    return true;
}

bool read_u32(const std::vector<uint8_t>& bytes, size_t& at, uint32_t& value) {
    uint8_t raw[4];
    if (!read_bytes(bytes, at, raw, sizeof(raw))) return false;
    value = (uint32_t) raw[0] | ((uint32_t) raw[1] << 8) | ((uint32_t) raw[2] << 16) | ((uint32_t) raw[3] << 24);
    return true;
}

template<class T> bool read_vector(const std::vector<uint8_t>& bytes, size_t& at, std::vector<T>& out, size_t count) {
    if (count > std::numeric_limits<size_t>::max() / sizeof(T)) return false;
    out.resize(count);
    return read_bytes(bytes, at, out.data(), count * sizeof(T));
}

// The fixture's own format (tools/exl3_fixture.py): the header, then the cases with their trellis, the
// scales, the bias, the input and the reference values.  Only the codebook 2 cases are used here.
bool load_fixture(const std::string& path, std::vector<FixtureCase>& cases, std::string& error) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        error = "cannot open fixture: " + path;
        return false;
    }
    const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    size_t at = 0;
    uint8_t magic[8];
    uint32_t version = 0, count = 0;
    if (!read_bytes(bytes, at, magic, sizeof(magic)) || std::memcmp(magic, "EXL3FX01", 8) != 0) {
        error = "fixture magic is not EXL3FX01";
        return false;
    }
    if (!read_u32(bytes, at, version) || !read_u32(bytes, at, count) || version != 1 || count == 0 || count > 10000) {
        error = "fixture header is invalid";
        return false;
    }
    for (uint32_t ci = 0; ci < count; ++ci) {
        FixtureCase c;
        std::vector<uint16_t> wgold;
        std::vector<float> ygold;
        if (!read_u32(bytes, at, c.bits) || !read_u32(bytes, at, c.codebook) || !read_u32(bytes, at, c.k) ||
            !read_u32(bytes, at, c.n) || c.bits < 1 || c.bits > 8 || c.codebook > 2 || c.k == 0 || c.n == 0 ||
            c.k % 16 != 0 || c.n % 16 != 0) {
            error = "fixture case metadata is invalid at case " + std::to_string(ci);
            return false;
        }
        const uint64_t tiles = (uint64_t) (c.k / 16) * (uint64_t) (c.n / 16);
        const uint64_t trellis_words = tiles * (uint64_t) (16 * c.bits);
        const uint64_t matrix_elements = (uint64_t) c.k * (uint64_t) c.n;
        if (trellis_words > std::numeric_limits<size_t>::max() || matrix_elements > std::numeric_limits<size_t>::max()) {
            error = "fixture case is too large at case " + std::to_string(ci);
            return false;
        }
        if (!read_vector(bytes, at, c.trellis, (size_t) trellis_words) ||
            !read_vector(bytes, at, c.suh, c.k) || !read_vector(bytes, at, c.svh, c.n) ||
            !read_vector(bytes, at, c.bias, c.n) || !read_vector(bytes, at, c.x, c.k) ||
            !read_vector(bytes, at, wgold, (size_t) matrix_elements) || !read_vector(bytes, at, ygold, c.n)) {
            error = "fixture is truncated at case " + std::to_string(ci);
            return false;
        }
        if (c.codebook == 2) cases.push_back(std::move(c));
    }
    return true;
}

// The same projection, once with the half form and once with the portable form: the values must be the
// same bits.  The switch is read at every launch, so both forms run in this process.
bool check_projection(sycl::queue& queue, strata::parity::Device& device, const FixtureCase& c, size_t index) {
    uint16_t* d_trellis = device.upload(c.trellis);
    uint16_t* d_suh = device.upload(c.suh);
    uint16_t* d_svh = device.upload(c.svh);
    uint16_t* d_bias = device.upload(c.bias);
    float* d_x = device.upload(c.x);
    float* d_y = device.alloc<float>(c.n);
    MatrixView view;
    view.trellis = d_trellis;
    view.suh = d_suh;
    view.svh = d_svh;
    view.bias = d_bias;
    view.k = c.k;
    view.n = c.n;
    view.bits = (int) c.bits;
    view.codebook = (int) c.codebook;
    const size_t scratch_elements = strata::kernels::exl3::exl3_gemv_scratch_elements(view);
    float* d_scratch = device.alloc<float>(scratch_elements);

    std::vector<float> portable(c.n), half(c.n);
    for (int form = 0; form < 2; ++form) {
        setenv("STRATA_EXL3_CB2_HALF_FMA", form == 0 ? "0" : "1", 1);
        strata::kernels::exl3::exl3_gemv_fp32(queue, view, d_x, d_y, d_scratch, scratch_elements, {}).wait_and_throw();
        strata::parity::copy_out(form == 0 ? portable.data() : half.data(), d_y, c.n);
    }
    unsetenv("STRATA_EXL3_CB2_HALF_FMA");
    size_t bad = 0;
    for (uint32_t j = 0; j < c.n; ++j)
        if (float_bits(portable[j]) != float_bits(half[j])) ++bad;
    std::printf("projection %2zu: bits %u [%ux%u] portable vs half: %s (%zu of %u values differ)\n", index, c.bits,
                c.k, c.n, bad == 0 ? "OK" : "FAIL", bad, c.n);
    return bad == 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::string fixture_path;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--fixture" && i + 1 < argc) {
            fixture_path = argv[++i];
        } else {
            std::fprintf(stderr, "usage: exl3_cb2_parity [--fixture PATH]\n");
            return 2;
        }
    }

    auto& queue = strata::core::Runtime::get().compute();
    strata::parity::Device device;
    const bool half_available = queue.get_device().has(sycl::aspect::fp16);
    std::printf("device fp16 arithmetic: %s\n", half_available ? "yes" : "no");

    int failures = 0;
    if (!check_byte_sums(queue, device, half_available)) ++failures;
    if (!check_byte_sum_forms(queue, device)) ++failures;
    if (!check_states(queue, device, half_available)) ++failures;
    if (!check_product_forms(queue, device)) ++failures;

    // The switch: "0" forces the portable form, anything else leaves the choice to the device.
    setenv("STRATA_EXL3_CB2_HALF_FMA", "0", 1);
    const bool forced_off = !strata::kernels::exl3::cb2_half_fma_enabled(queue);
    unsetenv("STRATA_EXL3_CB2_HALF_FMA");
    const bool device_choice = strata::kernels::exl3::cb2_half_fma_enabled(queue);
    std::printf("switch: STRATA_EXL3_CB2_HALF_FMA=0 -> %s, unset -> %s\n", forced_off ? "portable" : "half",
                device_choice ? "half" : "portable");
    if (!forced_off) {
        std::printf("switch: FAIL\n");
        ++failures;
    }
    if (device_choice != half_available) {
        std::printf("switch: FAIL (the device's choice does not follow its report)\n");
        ++failures;
    }

    if (!fixture_path.empty()) {
        std::vector<FixtureCase> cases;
        std::string error;
        if (!load_fixture(fixture_path, cases, error)) {
            std::fprintf(stderr, "exl3_cb2_parity: %s\n", error.c_str());
            return 2;
        }
        for (size_t i = 0; i < cases.size(); ++i)
            if (!check_projection(queue, device, cases[i], i)) ++failures;
    }

    std::printf("exl3_cb2_parity: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
