#ifndef INFERENCE_ENGINE_MEMORY_BUFFER_H_
#define INFERENCE_ENGINE_MEMORY_BUFFER_H_

/*
 * Raw memory buffer abstraction with ownership semantics and optional
 * debug canary guards for overflow detection.
 */

#include <cstddef>
#include <cstdint>

namespace inference_engine {
namespace core { class Allocator; }
namespace memory {

class Buffer {
public:
	Buffer() noexcept = default;

	Buffer(std::size_t size, std::size_t alignment = alignof(std::max_align_t),
		   core::Allocator* allocator = nullptr, bool use_canary = true);

	Buffer(void* data, std::size_t size, std::size_t alignment, bool owned = false,
		   bool use_canary = false) noexcept;

	Buffer(const Buffer& other);
	Buffer& operator=(const Buffer& other);

	Buffer(Buffer&& other) noexcept;
	Buffer& operator=(Buffer&& other) noexcept;

	~Buffer() noexcept;

	// Allocation / release
	bool allocate(std::size_t size, std::size_t alignment = alignof(std::max_align_t),
				  core::Allocator* allocator = nullptr, bool use_canary = true);
	void deallocate() noexcept;

	// Accessors
	void* data() noexcept { return data_; }
	const void* data() const noexcept { return data_; }
	std::size_t size() const noexcept { return size_; }
	std::size_t alignment() const noexcept { return alignment_; }
	bool owns_data() const noexcept { return owns_; }
	bool has_canary() const noexcept { return use_canary_; }

	// Debug guard validation
	bool validate_canary() const noexcept;

private:
	void* base_ = nullptr;    // pointer returned by allocator/raw new (includes canary)
	void* data_ = nullptr;    // user-visible data pointer (after canary prefix)
	std::size_t size_ = 0;
	std::size_t alignment_ = alignof(std::max_align_t);
	bool owns_ = false;
	bool use_canary_ = false;

	static constexpr std::uint32_t kCanaryValue = 0xDEADBEEF;
};

} // namespace memory
} // namespace inference_engine

#endif // INFERENCE_ENGINE_MEMORY_BUFFER_H_
