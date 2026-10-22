// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// The runtime that made an allocation, named for a pointer inside it.
//
// The engine carves its session arenas, its expert caches and its work buffers out of one
// allocation each, so most of the buffers it hands to a kernel are POINTERS INTO an allocation
// rather than its first address.  A context scan cannot tell one device's device USM from
// another's - get_pointer_type answers "device" for any of them - so a pointer into a block is
// named by the first runtime that claims it, which is the wrong device as soon as there are two.
// The multi-GPU EXL3 path refused its own stage's buffers for that reason.
//
// The check is strict on every device the engine can drive; with two or more devices it also
// states that a block's interior pointer is not named after the first runtime.

#include "strata/core/gpu.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/runtime.hpp"

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <vector>

namespace {

constexpr size_t kBlock = 1u << 20;
constexpr size_t kOffset = 4096;

bool names_the_making_runtime(int device) {
    const strata::core::OnDevice on(device);
    void* block = strata::gpu::alloc_device(kBlock);
    if (block == nullptr) {
        std::cerr << "device " << device << ": the device allocation failed\n";
        return false;
    }
    auto* interior = static_cast<char*>(block) + kOffset;
    strata::core::Runtime* expected = &strata::core::Runtime::at(device);
    const bool base_ok = strata::core::Runtime::owner(block) == expected;
    const bool interior_ok = strata::core::Runtime::owner(interior) == expected;
    void* host = strata::gpu::alloc_host(kBlock);
    auto* host_interior = static_cast<char*>(host) + kOffset;
    const bool host_ok = host != nullptr && strata::core::Runtime::owner(host_interior) == expected;
    if (host != nullptr) strata::gpu::free(host);
    strata::gpu::free(block);
    if (!base_ok || !interior_ok || !host_ok) {
        std::cerr << "device " << device << ": the allocation's owner is base=" << (base_ok ? "right" : "wrong")
                  << " interior=" << (interior_ok ? "right" : "wrong") << " host=" << (host_ok ? "right" : "wrong")
                  << '\n';
        return false;
    }
    return true;
}

}  // namespace

int main() {
    const int devices = strata::core::Runtime::count();
    if (devices == 0) return 77;
    for (int device = 0; device < devices; ++device)
        if (!names_the_making_runtime(device)) return 1;
    if (devices > 1) {
        // The first runtime claims every device pointer a context scan sees, so a block on the
        // second device would be named after the first one without the record.
        for (int device = 1; device < devices; ++device) {
            const strata::core::OnDevice on(device);
            void* block = strata::gpu::alloc_device(kBlock);
            if (block == nullptr) {
                std::cerr << "device " << device << ": the device allocation failed\n";
                return 1;
            }
            auto* interior = static_cast<char*>(block) + kOffset;
            strata::core::Runtime* owner = strata::core::Runtime::owner(interior);
            const bool named = owner == &strata::core::Runtime::at(device);
            strata::gpu::free(block);
            if (!named) {
                std::cerr << "device " << device << ": a pointer inside its own block was named after device "
                          << (owner ? owner->ordinal() : -1) << '\n';
                return 1;
            }
        }
    }
    std::printf("runtime ownership checks: OK (%d device%s)\n", devices, devices == 1 ? "" : "s");
    return 0;
}
