#ifndef INFERENCE_ENGINE_MEMORY_ALLOCATOR_H_
#define INFERENCE_ENGINE_MEMORY_ALLOCATOR_H_

/*
 * High-level allocator interface for the inference engine.
 *
 * Design goals:
 * - Keep a minimal virtual interface (`allocate`/`deallocate`) used throughout the codebase.
 * - Allow different backends (system malloc/free, Arena, etc.).
 * - Provide alignment controls and optional allocation tracking for debugging.
 */

#include <cstddef>
#include <cstdint>
#include <memory>

#include "inference_engine/memory/arena.h"

namespace inference_engine {
namespace core {

    struct AllocationStats {
        std::size_t allocations = 0;
        std::size_t frees = 0;
        std::size_t bytes_allocated = 0;
        std::size_t bytes_freed = 0;
        std::size_t live_allocations = 0;
        std::size_t live_bytes = 0;
        std::size_t peak_live_bytes = 0;
    };

    struct AllocatorConfig {
        std::size_t alignment = alignof(std::max_align_t);
        bool track_allocations = false;
    };

    /*
     * Abstract allocator interface.
     * Implementations can provide different allocation strategies (arena, pool, etc).
     */
    class Allocator {
    public:
        virtual ~Allocator() noexcept = default;

        /*
         * Allocate memory of the specified size in bytes.
         * Returns nullptr if allocation fails.
         */
        virtual void* allocate(int64_t size_bytes) = 0;

        /*
         * Deallocate previously allocated memory.
         * ptr should be a pointer returned by allocate().
         */
        virtual void deallocate(void* ptr) noexcept = 0;

        /*
         * Reallocate memory to a new size.
         * Preserves content up to min(old_size, new_size).
         * Returns nullptr if reallocation fails.
         */
        // Optional: backends can override if they can preserve data.
        virtual void* reallocate(void* /*ptr*/, int64_t /*new_size_bytes*/) { return nullptr; }

        /*
         * Get alignment requirement in bytes.
         */
        // Default alignment expectation for allocations from this allocator.
        virtual std::size_t alignment() const noexcept { return alignof(std::max_align_t); }

        // Optional aligned allocation API; default delegates to allocate().
        // Backends can override to honor the per-allocation alignment.
        virtual void* allocate_aligned(std::size_t size_bytes, std::size_t /*alignment_bytes*/) {
            return allocate(static_cast<int64_t>(size_bytes));
        }

        /*
         * Check if this allocator owns the given pointer.
         */
        virtual bool owns(const void* /*ptr*/) const noexcept { return true; }

        // Debugging / introspection
        virtual bool tracking_enabled() const noexcept { return false; }
        virtual AllocationStats stats() const noexcept { return {}; }
        virtual void reset_stats() noexcept {}
    };

    // ==================== Backends ====================

    // System allocator implementation backed by aligned malloc/free.
    class SystemAllocator final : public Allocator {
    public:
        explicit SystemAllocator(AllocatorConfig config = {}) noexcept;
        ~SystemAllocator() noexcept override;

        void* allocate(int64_t size_bytes) override;
        void deallocate(void* ptr) noexcept override;
        void* reallocate(void* ptr, int64_t new_size_bytes) override;

        std::size_t alignment() const noexcept override { return alignment_; }
        void* allocate_aligned(std::size_t size_bytes, std::size_t alignment_bytes) override;
        bool owns(const void* ptr) const noexcept override;

        bool tracking_enabled() const noexcept override { return track_allocations_; }
        AllocationStats stats() const noexcept override;
        void reset_stats() noexcept override;

    private:
        std::size_t alignment_ = alignof(std::max_align_t);
        bool track_allocations_ = false;

        // Allocation tracking is implemented in allocator.cpp (opaque here).
        struct TrackingState;
        std::unique_ptr<TrackingState> tracking_;
    };

    // Arena-backed allocator. Individual deallocations are no-ops; call reset().
    class ArenaAllocator final : public Allocator {
    public:
        explicit ArenaAllocator(std::size_t arena_capacity_bytes,
                               std::size_t arena_base_alignment = alignof(std::max_align_t),
                               AllocatorConfig config = {}) noexcept;
        ~ArenaAllocator() noexcept override;

        void* allocate(int64_t size_bytes) override;
        void deallocate(void* ptr) noexcept override;

        std::size_t alignment() const noexcept override { return alignment_; }
        void* allocate_aligned(std::size_t size_bytes, std::size_t alignment_bytes) override;
        bool owns(const void* ptr) const noexcept override;

        void reset() noexcept;
        const memory::Arena& arena() const noexcept { return arena_; }
        memory::Arena& arena() noexcept { return arena_; }

        bool tracking_enabled() const noexcept override { return track_allocations_; }
        AllocationStats stats() const noexcept override;
        void reset_stats() noexcept override;

    private:
        memory::Arena arena_;
        std::size_t alignment_ = alignof(std::max_align_t);
        bool track_allocations_ = false;

        struct TrackingState;
        std::unique_ptr<TrackingState> tracking_;
    };

    // ==================== Factory helpers ====================

    std::unique_ptr<Allocator> make_system_allocator(AllocatorConfig config = {});
    std::unique_ptr<Allocator> make_arena_allocator(std::size_t arena_capacity_bytes,
                                                    std::size_t arena_base_alignment = alignof(std::max_align_t),
                                                    AllocatorConfig config = {});

} // namespace core
} // namespace inference_engine

#endif // INFERENCE_ENGINE_MEMORY_ALLOCATOR_H_
