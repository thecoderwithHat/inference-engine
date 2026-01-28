#include "inference_engine/memory/buffer.h"
#include "inference_engine/memory/allocator.h"

#include <cstdlib>
#include <cstring>
#include <new>

namespace inference_engine {
namespace memory {

namespace {

// Platform-aligned allocation helper (aligned to `alignment`).
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

Buffer::Buffer(std::size_t size, std::size_t alignment, core::Allocator* allocator, bool use_canary)
	: alignment_(alignment), owns_(true), use_canary_(use_canary) {
	allocate(size, alignment, allocator, use_canary);
}

Buffer::Buffer(void* data, std::size_t size, std::size_t alignment, bool owned, bool use_canary) noexcept
	: base_(data), data_(data), size_(size), alignment_(alignment), owns_(owned), use_canary_(use_canary) {}

Buffer::Buffer(const Buffer& other) {
	if (other.data_ && other.size_ > 0) {
		allocate(other.size_, other.alignment_, nullptr, other.use_canary_);
		if (data_) {
			std::memcpy(data_, other.data_, other.size_);
		}
	}
}

Buffer& Buffer::operator=(const Buffer& other) {
	if (this == &other) return *this;
	if (owns_) {
		deallocate();
	}
	if (other.data_ && other.size_ > 0) {
		allocate(other.size_, other.alignment_, nullptr, other.use_canary_);
		if (data_) {
			std::memcpy(data_, other.data_, other.size_);
		}
	}
	return *this;
}

Buffer::Buffer(Buffer&& other) noexcept {
	base_ = other.base_;
	data_ = other.data_;
	size_ = other.size_;
	alignment_ = other.alignment_;
	owns_ = other.owns_;
	use_canary_ = other.use_canary_;

	other.base_ = nullptr;
	other.data_ = nullptr;
	other.size_ = 0;
	other.owns_ = false;
	other.use_canary_ = false;
}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
	if (this == &other) return *this;

	if (owns_) {
		deallocate();
	}

	base_ = other.base_;
	data_ = other.data_;
	size_ = other.size_;
	alignment_ = other.alignment_;
	owns_ = other.owns_;
	use_canary_ = other.use_canary_;

	other.base_ = nullptr;
	other.data_ = nullptr;
	other.size_ = 0;
	other.owns_ = false;
	other.use_canary_ = false;
	return *this;
}

Buffer::~Buffer() noexcept {
	if (owns_) {
		deallocate();
	}
}

bool Buffer::allocate(std::size_t size, std::size_t alignment, core::Allocator* allocator, bool use_canary) {
	if (owns_) {
		deallocate();
	}

	alignment_ = alignment;
	size_ = size;
	use_canary_ = use_canary;

	std::size_t total_size = size;
	if (use_canary_) {
		// prepend + append canary (4 bytes each)
		total_size += sizeof(kCanaryValue) * 2;
	}

	if (allocator) {
		base_ = allocator->allocate(static_cast<int64_t>(total_size));
	} else {
		base_ = allocate_aligned(total_size, alignment_);
	}

	if (!base_) {
		data_ = nullptr;
		size_ = 0;
		return false;
	}

	owns_ = true;

	if (use_canary_) {
		// layout: [canary][data...][canary]
		auto* prefix = reinterpret_cast<std::uint32_t*>(base_);
		*prefix = kCanaryValue;
		data_ = reinterpret_cast<void*>(prefix + 1);
		auto* suffix = reinterpret_cast<std::uint32_t*>(reinterpret_cast<std::uint8_t*>(data_) + size_);
		*suffix = kCanaryValue;
	} else {
		data_ = base_;
	}

	return true;
}

void Buffer::deallocate() noexcept {
	if (!base_ || !owns_) {
		base_ = nullptr;
		data_ = nullptr;
		size_ = 0;
		return;
	}

	if (use_canary_) {
		// wipe canaries for clarity
		auto* prefix = reinterpret_cast<std::uint32_t*>(base_);
		*prefix = 0;
		auto* suffix = reinterpret_cast<std::uint32_t*>(reinterpret_cast<std::uint8_t*>(data_) + size_);
		*suffix = 0;
	}

	free_aligned(base_);

	base_ = nullptr;
	data_ = nullptr;
	size_ = 0;
	owns_ = false;
}

bool Buffer::validate_canary() const noexcept {
	if (!use_canary_ || !base_ || size_ == 0) {
		return true;
	}
	auto* prefix = reinterpret_cast<const std::uint32_t*>(base_);
	auto* suffix = reinterpret_cast<const std::uint32_t*>(reinterpret_cast<const std::uint8_t*>(data_) + size_);
	return *prefix == kCanaryValue && *suffix == kCanaryValue;
}

} // namespace memory
} // namespace inference_engine
