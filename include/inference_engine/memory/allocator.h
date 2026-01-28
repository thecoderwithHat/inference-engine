#ifndef INFERENCE_ENGINE_MEMORY_ALLOCATOR_H_
#define INFERENCE_ENGINE_MEMORY_ALLOCATOR_H_

/*
 * Memory allocator interface for the inference engine.
 * Provides abstraction for memory allocation and deallocation.
 */

#include <cstdint>
#include <cstddef>

namespace inference_engine {
namespace core {

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
        virtual void* reallocate(void* ptr, int64_t new_size_bytes) {
            // Default implementation: allocate new, copy, deallocate old
            // Subclasses can override for more efficient implementations
            return nullptr;
        }

        /*
         * Get alignment requirement in bytes.
         */
        virtual std::size_t alignment() const noexcept { return 16; }

        /*
         * Check if this allocator owns the given pointer.
         */
        virtual bool owns(const void* ptr) const noexcept { return true; }
    };

} // namespace core
} // namespace inference_engine

#endif // INFERENCE_ENGINE_MEMORY_ALLOCATOR_H_
