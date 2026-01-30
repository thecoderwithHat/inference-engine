#ifndef INFERENCE_ENGINE_MEMORY_ARENA_H_
#define INFERENCE_ENGINE_MEMORY_ARENA_H_

/*
 * Fast bump allocator (arena) for inference workloads.
 *
 * - Allocations are linear and extremely fast (bump pointer).
 * - Individual frees are not supported; call reset() to reuse the whole arena.
 * - Not thread-safe: use one Arena per thread (thread-local) or guard externally.
 */

#include <cstddef>

namespace inference_engine {
namespace memory {

class Arena {
public:
    struct Stats {
        std::size_t allocations = 0;
        std::size_t peak_used_bytes = 0;
    };

    // Creates an arena with a pre-allocated buffer of `capacity_bytes`.
    // `base_alignment` controls the alignment of the backing allocation.
    explicit Arena(std::size_t capacity_bytes,
                  std::size_t base_alignment = alignof(std::max_align_t)) noexcept;

    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;

    Arena(Arena&& other) noexcept;
    Arena& operator=(Arena&& other) noexcept;

    ~Arena() noexcept;

    // Bump allocation; returns nullptr on out-of-memory or invalid alignment.
    // If `alignment` is 0, defaults to alignof(std::max_align_t).
    void* allocate(std::size_t size_bytes,
                   std::size_t alignment = alignof(std::max_align_t)) noexcept;

    // Resets the bump pointer without freeing backing memory.
    // Also resets per-cycle stats (allocations, peak usage).
    void reset() noexcept;

    // Capacity and usage
    std::size_t capacity() const noexcept { return capacity_bytes_; }
    std::size_t used() const noexcept { return used_bytes_; }
    std::size_t remaining() const noexcept { return capacity_bytes_ - used_bytes_; }

    Stats stats() const noexcept { return stats_; }

    // Returns true if `ptr` lies within the backing arena buffer.
    // Note: this does not guarantee `ptr` refers to the start of a live allocation.
    bool owns(const void* ptr) const noexcept;

private:
    static bool is_power_of_two(std::size_t x) noexcept;
    static std::size_t align_up(std::size_t value, std::size_t alignment) noexcept;

    void* base_ = nullptr;
    std::size_t capacity_bytes_ = 0;
    std::size_t used_bytes_ = 0;
    std::size_t base_alignment_ = alignof(std::max_align_t);
    Stats stats_{};
};

} // namespace memory
} // namespace inference_engine

#endif // INFERENCE_ENGINE_MEMORY_ARENA_H_
