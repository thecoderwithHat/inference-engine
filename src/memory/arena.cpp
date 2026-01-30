#include "inference_engine/memory/arena.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>

namespace inference_engine {
namespace memory {

namespace {

inline void* allocate_aligned(std::size_t size, std::size_t alignment) noexcept {
#if defined(_MSC_VER)
    return _aligned_malloc(size, alignment);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return nullptr;
    }
    return ptr;
#endif
}

inline void free_aligned(void* ptr) noexcept {
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

} // namespace

bool Arena::is_power_of_two(std::size_t x) noexcept {
    return x != 0 && (x & (x - 1)) == 0;
}

std::size_t Arena::align_up(std::size_t value, std::size_t alignment) noexcept {
    return (value + (alignment - 1)) & ~(alignment - 1);
}

Arena::Arena(std::size_t capacity_bytes, std::size_t base_alignment) noexcept
    : capacity_bytes_(capacity_bytes), base_alignment_(base_alignment) {
    if (base_alignment_ == 0) {
        base_alignment_ = alignof(std::max_align_t);
    }
    // posix_memalign requires power-of-two and multiple of sizeof(void*)
    if (!is_power_of_two(base_alignment_) || base_alignment_ < sizeof(void*)) {
        base_alignment_ = alignof(std::max_align_t);
    }

    if (capacity_bytes_ == 0) {
        base_ = nullptr;
        return;
    }

    base_ = allocate_aligned(capacity_bytes_, base_alignment_);
    if (!base_) {
        capacity_bytes_ = 0;
        used_bytes_ = 0;
        stats_ = {};
    }
}

Arena::Arena(Arena&& other) noexcept {
    base_ = other.base_;
    capacity_bytes_ = other.capacity_bytes_;
    used_bytes_ = other.used_bytes_;
    base_alignment_ = other.base_alignment_;
    stats_ = other.stats_;

    other.base_ = nullptr;
    other.capacity_bytes_ = 0;
    other.used_bytes_ = 0;
    other.stats_ = {};
}

Arena& Arena::operator=(Arena&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    if (base_) {
        free_aligned(base_);
    }

    base_ = other.base_;
    capacity_bytes_ = other.capacity_bytes_;
    used_bytes_ = other.used_bytes_;
    base_alignment_ = other.base_alignment_;
    stats_ = other.stats_;

    other.base_ = nullptr;
    other.capacity_bytes_ = 0;
    other.used_bytes_ = 0;
    other.stats_ = {};

    return *this;
}

Arena::~Arena() noexcept {
    if (base_) {
        free_aligned(base_);
        base_ = nullptr;
    }
}

void* Arena::allocate(std::size_t size_bytes, std::size_t alignment) noexcept {
    if (!base_ || capacity_bytes_ == 0) {
        return nullptr;
    }

    if (alignment == 0) {
        alignment = alignof(std::max_align_t);
    }
    if (!is_power_of_two(alignment)) {
        return nullptr;
    }

    const auto base_addr = reinterpret_cast<std::uintptr_t>(base_);
    const auto current_addr = base_addr + used_bytes_;
    const auto aligned_addr = static_cast<std::uintptr_t>(align_up(static_cast<std::size_t>(current_addr), alignment));

    const std::size_t aligned_offset = static_cast<std::size_t>(aligned_addr - base_addr);
    if (aligned_offset > capacity_bytes_) {
        return nullptr;
    }

    // Avoid overflow
    if (size_bytes > capacity_bytes_ - aligned_offset) {
        return nullptr;
    }

    used_bytes_ = aligned_offset + size_bytes;
    stats_.allocations += 1;
    stats_.peak_used_bytes = std::max(stats_.peak_used_bytes, used_bytes_);
    return reinterpret_cast<void*>(aligned_addr);
}

void Arena::reset() noexcept {
    used_bytes_ = 0;
    stats_ = {};
}

bool Arena::owns(const void* ptr) const noexcept {
    if (!base_ || !ptr || capacity_bytes_ == 0) {
        return false;
    }
    const auto base_addr = reinterpret_cast<std::uintptr_t>(base_);
    const auto ptr_addr = reinterpret_cast<std::uintptr_t>(ptr);
    return ptr_addr >= base_addr && ptr_addr < (base_addr + capacity_bytes_);
}

} // namespace memory
} // namespace inference_engine
