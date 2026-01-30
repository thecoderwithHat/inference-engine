#include "inference_engine/memory/allocator.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <unordered_map>

namespace inference_engine {
namespace core {

namespace {

inline bool is_power_of_two(std::size_t x) noexcept {
	return x != 0 && (x & (x - 1)) == 0;
}

inline std::size_t normalize_alignment(std::size_t alignment) noexcept {
	if (alignment == 0) {
		return alignof(std::max_align_t);
	}
	if (!is_power_of_two(alignment) || alignment < sizeof(void*)) {
		return alignof(std::max_align_t);
	}
	return alignment;
}

inline void* allocate_aligned_system(std::size_t size, std::size_t alignment) noexcept {
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

inline void free_aligned_system(void* ptr) noexcept {
#if defined(_MSC_VER)
	_aligned_free(ptr);
#else
	std::free(ptr);
#endif
}

inline void update_peak(AllocationStats& s) noexcept {
	s.peak_live_bytes = std::max(s.peak_live_bytes, s.live_bytes);
}

} // namespace

// ==================== SystemAllocator ====================

struct SystemAllocator::TrackingState {
	mutable std::mutex mu;
	std::unordered_map<const void*, std::size_t> live_sizes;
	AllocationStats stats;
};

SystemAllocator::~SystemAllocator() noexcept = default;

SystemAllocator::SystemAllocator(AllocatorConfig config) noexcept
	: alignment_(normalize_alignment(config.alignment)), track_allocations_(config.track_allocations) {
	if (track_allocations_) {
		tracking_ = std::make_unique<TrackingState>();
	}
}

void* SystemAllocator::allocate(int64_t size_bytes) {
	if (size_bytes <= 0) {
		return nullptr;
	}
	return allocate_aligned(static_cast<std::size_t>(size_bytes), alignment_);
}

void* SystemAllocator::allocate_aligned(std::size_t size_bytes, std::size_t alignment_bytes) {
	if (size_bytes == 0) {
		return nullptr;
	}

	const std::size_t alignment = normalize_alignment(alignment_bytes == 0 ? alignment_ : alignment_bytes);
	void* ptr = allocate_aligned_system(size_bytes, alignment);
	if (!ptr) {
		return nullptr;
	}

	if (track_allocations_ && tracking_) {
		std::lock_guard<std::mutex> lock(tracking_->mu);
		tracking_->live_sizes[ptr] = size_bytes;
		tracking_->stats.allocations += 1;
		tracking_->stats.bytes_allocated += size_bytes;
		tracking_->stats.live_allocations += 1;
		tracking_->stats.live_bytes += size_bytes;
		update_peak(tracking_->stats);
	}

	return ptr;
}

void SystemAllocator::deallocate(void* ptr) noexcept {
	if (!ptr) {
		return;
	}

	if (track_allocations_ && tracking_) {
		std::lock_guard<std::mutex> lock(tracking_->mu);
		const auto it = tracking_->live_sizes.find(ptr);
		if (it != tracking_->live_sizes.end()) {
			const std::size_t size = it->second;
			tracking_->live_sizes.erase(it);

			tracking_->stats.frees += 1;
			tracking_->stats.bytes_freed += size;
			if (tracking_->stats.live_allocations > 0) {
				tracking_->stats.live_allocations -= 1;
			}
			if (tracking_->stats.live_bytes >= size) {
				tracking_->stats.live_bytes -= size;
			} else {
				tracking_->stats.live_bytes = 0;
			}
		} else {
			// Unknown pointer: still free it, but don't try to account bytes.
			tracking_->stats.frees += 1;
		}
	}

	free_aligned_system(ptr);
}

void* SystemAllocator::reallocate(void* ptr, int64_t new_size_bytes) {
	if (new_size_bytes <= 0) {
		deallocate(ptr);
		return nullptr;
	}

	// If we can't determine old size, we can't safely preserve content.
	std::size_t old_size = 0;
	if (ptr && track_allocations_ && tracking_) {
		std::lock_guard<std::mutex> lock(tracking_->mu);
		const auto it = tracking_->live_sizes.find(ptr);
		if (it != tracking_->live_sizes.end()) {
			old_size = it->second;
		}
	}

	void* new_ptr = allocate(static_cast<int64_t>(new_size_bytes));
	if (!new_ptr) {
		return nullptr;
	}

	if (ptr && old_size > 0) {
		const std::size_t copy_n = std::min<std::size_t>(old_size, static_cast<std::size_t>(new_size_bytes));
		std::memcpy(new_ptr, ptr, copy_n);
	}

	deallocate(ptr);
	return new_ptr;
}

bool SystemAllocator::owns(const void* ptr) const noexcept {
	if (!ptr) {
		return false;
	}
	if (track_allocations_ && tracking_) {
		std::lock_guard<std::mutex> lock(tracking_->mu);
		return tracking_->live_sizes.find(ptr) != tracking_->live_sizes.end();
	}
	return true;
}

AllocationStats SystemAllocator::stats() const noexcept {
	if (!track_allocations_ || !tracking_) {
		return {};
	}
	std::lock_guard<std::mutex> lock(tracking_->mu);
	return tracking_->stats;
}

void SystemAllocator::reset_stats() noexcept {
	if (!track_allocations_ || !tracking_) {
		return;
	}
	std::lock_guard<std::mutex> lock(tracking_->mu);
	tracking_->stats = {};
}

// ==================== ArenaAllocator ====================

struct ArenaAllocator::TrackingState {
	mutable std::mutex mu;
	std::unordered_map<const void*, std::size_t> live_sizes;
	AllocationStats stats;
};

ArenaAllocator::~ArenaAllocator() noexcept = default;

ArenaAllocator::ArenaAllocator(std::size_t arena_capacity_bytes,
							   std::size_t arena_base_alignment,
							   AllocatorConfig config) noexcept
	: arena_(arena_capacity_bytes, arena_base_alignment),
	  alignment_(normalize_alignment(config.alignment)),
	  track_allocations_(config.track_allocations) {
	if (track_allocations_) {
		tracking_ = std::make_unique<TrackingState>();
	}
}

void* ArenaAllocator::allocate(int64_t size_bytes) {
	if (size_bytes <= 0) {
		return nullptr;
	}
	return allocate_aligned(static_cast<std::size_t>(size_bytes), alignment_);
}

void* ArenaAllocator::allocate_aligned(std::size_t size_bytes, std::size_t alignment_bytes) {
	if (size_bytes == 0) {
		return nullptr;
	}
	const std::size_t alignment = normalize_alignment(alignment_bytes == 0 ? alignment_ : alignment_bytes);
	void* ptr = arena_.allocate(size_bytes, alignment);
	if (!ptr) {
		return nullptr;
	}

	if (track_allocations_ && tracking_) {
		std::lock_guard<std::mutex> lock(tracking_->mu);
		tracking_->live_sizes[ptr] = size_bytes;
		tracking_->stats.allocations += 1;
		tracking_->stats.bytes_allocated += size_bytes;
		tracking_->stats.live_allocations += 1;
		tracking_->stats.live_bytes += size_bytes;
		update_peak(tracking_->stats);
	}

	return ptr;
}

void ArenaAllocator::deallocate(void* ptr) noexcept {
	// Arena allocations are freed en-masse via reset().
	if (!ptr) {
		return;
	}

	if (track_allocations_ && tracking_) {
		std::lock_guard<std::mutex> lock(tracking_->mu);
		const auto it = tracking_->live_sizes.find(ptr);
		if (it != tracking_->live_sizes.end()) {
			const std::size_t size = it->second;
			tracking_->live_sizes.erase(it);

			tracking_->stats.frees += 1;
			tracking_->stats.bytes_freed += size;
			if (tracking_->stats.live_allocations > 0) {
				tracking_->stats.live_allocations -= 1;
			}
			if (tracking_->stats.live_bytes >= size) {
				tracking_->stats.live_bytes -= size;
			} else {
				tracking_->stats.live_bytes = 0;
			}
		} else {
			tracking_->stats.frees += 1;
		}
	}
}

bool ArenaAllocator::owns(const void* ptr) const noexcept {
	if (!ptr) {
		return false;
	}
	if (track_allocations_ && tracking_) {
		std::lock_guard<std::mutex> lock(tracking_->mu);
		return tracking_->live_sizes.find(ptr) != tracking_->live_sizes.end();
	}
	return arena_.owns(ptr);
}

void ArenaAllocator::reset() noexcept {
	arena_.reset();
	if (track_allocations_ && tracking_) {
		std::lock_guard<std::mutex> lock(tracking_->mu);
		tracking_->live_sizes.clear();
		tracking_->stats.live_allocations = 0;
		tracking_->stats.live_bytes = 0;
	}
}

AllocationStats ArenaAllocator::stats() const noexcept {
	if (!track_allocations_ || !tracking_) {
		return {};
	}
	std::lock_guard<std::mutex> lock(tracking_->mu);
	return tracking_->stats;
}

void ArenaAllocator::reset_stats() noexcept {
	if (!track_allocations_ || !tracking_) {
		return;
	}
	std::lock_guard<std::mutex> lock(tracking_->mu);
	tracking_->stats = {};
}

// ==================== Factories ====================

std::unique_ptr<Allocator> make_system_allocator(AllocatorConfig config) {
	return std::make_unique<SystemAllocator>(config);
}

std::unique_ptr<Allocator> make_arena_allocator(std::size_t arena_capacity_bytes,
												std::size_t arena_base_alignment,
												AllocatorConfig config) {
	return std::make_unique<ArenaAllocator>(arena_capacity_bytes, arena_base_alignment, config);
}

} // namespace core
} // namespace inference_engine

