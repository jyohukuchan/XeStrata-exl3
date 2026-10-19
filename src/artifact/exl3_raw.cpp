// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "strata/artifact/exl3_raw.hpp"

#include "strata/kernels/f16_bits.hpp"

#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace strata::artifact::exl3 {
namespace {

size_t dtype_bytes(const std::string& dtype) {
    if (dtype == "F16" || dtype == "BF16") return 2;
    if (dtype == "F32") return 4;
    throw std::invalid_argument("raw dtype is not F16/BF16/F32");
}

size_t elements(const std::vector<uint64_t>& shape, size_t width) {
    const size_t limit = static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()) / width;
    size_t count = 1;
    for (uint64_t dim : shape) {
        if (dim == 0 || dim > limit / count) throw std::invalid_argument("raw shape exceeds the address range");
        count *= static_cast<size_t>(dim);
    }
    return count;
}

float read_value(const uint8_t* data, const std::string& dtype) {
    uint32_t bits = static_cast<uint32_t>(data[0]) | (static_cast<uint32_t>(data[1]) << 8);
    if (dtype == "F16") return strata::kernels::f32_from_f16(static_cast<uint16_t>(bits));
    if (dtype == "BF16") bits <<= 16;
    else bits |= (static_cast<uint32_t>(data[2]) << 16) | (static_cast<uint32_t>(data[3]) << 24);
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

size_t validate(const RawTensor& source) {
    const size_t width = dtype_bytes(source.dtype);
    const size_t count = elements(source.shape, width);
    if (source.data.size() != count * width) throw std::invalid_argument("raw payload length does not match shape");
    return count;
}

}  // namespace

bool remap_raw(const RawTensor& source, const std::vector<std::vector<int64_t>>& axes,
               ValueTransform transform, RawTensor& destination, std::string& err) {
    err.clear();
    try {
        (void) validate(source);
        if (transform != ValueTransform::identity && transform != ValueTransform::add_one &&
            transform != ValueTransform::negative_exp) throw std::invalid_argument("unknown raw value transform");
        const size_t rank = source.shape.size(), width = dtype_bytes(source.dtype);
        if (!axes.empty() && axes.size() != rank) throw std::invalid_argument("raw axis map has the wrong rank");
        RawTensor result;
        result.dtype = transform == ValueTransform::identity ? source.dtype : "F32";
        result.shape = source.shape;
        bool identity_axes = true;
        for (size_t axis = 0; axis < axes.size(); ++axis) {
            const auto& map = axes[axis];
            if (map.empty()) continue;
            if (map.size() > source.shape[axis]) throw std::invalid_argument("raw axis map is too long");
            std::vector<uint8_t> seen(static_cast<size_t>(source.shape[axis]), 0);
            result.shape[axis] = map.size();
            identity_axes &= map.size() == source.shape[axis];
            for (size_t i = 0; i < map.size(); ++i) {
                if (map[i] < 0 || static_cast<uint64_t>(map[i]) >= source.shape[axis] || seen[map[i]])
                    throw std::invalid_argument("raw axis indices must be unique and in range");
                seen[map[i]] = 1;
                identity_axes &= static_cast<uint64_t>(map[i]) == i;
            }
        }
        const size_t output_width = dtype_bytes(result.dtype);
        const size_t count = elements(result.shape, output_width);
        result.data.resize(count * output_width);
        if (identity_axes && transform == ValueTransform::identity) {
            result.data = source.data;
        } else {
            std::vector<size_t> source_stride(rank, 1);
            for (size_t axis = rank; axis > 1; --axis)
                source_stride[axis - 2] = source_stride[axis - 1] * static_cast<size_t>(source.shape[axis - 1]);
            for (size_t flat = 0; flat < count; ++flat) {
                size_t rest = flat, source_flat = 0;
                for (size_t a = rank; a > 0; --a) {
                    const size_t axis = a - 1;
                    const size_t index = rest % static_cast<size_t>(result.shape[axis]);
                    rest /= static_cast<size_t>(result.shape[axis]);
                    const size_t source_index = axes.empty() || axes[axis].empty()
                        ? index : static_cast<size_t>(axes[axis][index]);
                    source_flat += source_index * source_stride[axis];
                }
                const uint8_t* value = source.data.data() + source_flat * width;
                uint8_t* out = result.data.data() + flat * output_width;
                if (transform == ValueTransform::identity) std::memcpy(out, value, width);
                else {
                    const float x = read_value(value, source.dtype);
                    const float y = transform == ValueTransform::add_one ? x + 1.0f : -std::exp(x);
                    std::memcpy(out, &y, sizeof(y));
                }
            }
        }
        destination = std::move(result);
        return true;
    } catch (const std::exception& e) {
        err = std::string("EXL3 raw materializer: ") + e.what();
        return false;
    }
}

bool materialize_raw(const Archive& archive, const TensorDescriptor& descriptor,
                     const std::vector<std::vector<int64_t>>& axes, ValueTransform transform,
                     RawTensor& destination, std::string& err) {
    err.clear();
    try {
        const size_t count = elements(descriptor.shape, dtype_bytes(descriptor.dtype));
        if (descriptor.bytes != count * dtype_bytes(descriptor.dtype))
            throw std::invalid_argument("raw descriptor length does not match shape");
        RawTensor source{descriptor.dtype, descriptor.shape, {}};
        archive.read_tensor(descriptor, source.data);
        return remap_raw(source, axes, transform, destination, err);
    } catch (const std::exception& e) {
        err = std::string("EXL3 raw materializer: ") + e.what();
        return false;
    }
}

bool raw_to_f32(const RawTensor& source, std::vector<float>& destination, std::string& err) {
    err.clear();
    try {
        const size_t count = validate(source);
        if (count > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()) / sizeof(float))
            throw std::invalid_argument("widened raw tensor exceeds the address range");
        const size_t width = dtype_bytes(source.dtype);
        std::vector<float> result(count);
        for (size_t i = 0; i < count; ++i) result[i] = read_value(source.data.data() + i * width, source.dtype);
        destination = std::move(result);
        return true;
    } catch (const std::exception& e) {
        err = std::string("EXL3 raw materializer: ") + e.what();
        return false;
    }
}

}  // namespace strata::artifact::exl3
