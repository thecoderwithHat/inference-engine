#include <gtest/gtest.h>

#include "inference_engine/memory/allocator.h"

#include <cstring>
#include <cstdint>
#include <limits>
#include <thread>
#include <vector>

using inference_engine::core::Allocator;
using inference_engine::core::AllocatorConfig;
using inference_engine::core::ArenaAllocator;
using inference_engine::core::SystemAllocator;

TEST(AllocatorTest, SystemAllocatorBasicAllocFreeTracking) {
	SystemAllocator alloc(AllocatorConfig{64, true});

	void* p = alloc.allocate(128);
	ASSERT_NE(p, nullptr);
	EXPECT_EQ(reinterpret_cast<std::uintptr_t>(p) % 64u, 0u);

	auto s = alloc.stats();
	EXPECT_EQ(s.allocations, 1u);
	EXPECT_EQ(s.live_allocations, 1u);
	EXPECT_EQ(s.live_bytes, 128u);
	EXPECT_GE(s.peak_live_bytes, s.live_bytes);

	EXPECT_TRUE(alloc.owns(p));
	alloc.deallocate(p);
	EXPECT_FALSE(alloc.owns(p));

	s = alloc.stats();
	EXPECT_EQ(s.frees, 1u);
	EXPECT_EQ(s.live_allocations, 0u);
	EXPECT_EQ(s.live_bytes, 0u);
}

TEST(AllocatorTest, SystemAllocatorFailureModes) {
	SystemAllocator alloc(AllocatorConfig{alignof(std::max_align_t), true});

	EXPECT_EQ(alloc.allocate(0), nullptr);
	EXPECT_EQ(alloc.allocate(-1), nullptr);
	alloc.deallocate(nullptr);

	// This may or may not fail depending on platform overcommit, so just sanity-check it doesn't crash.
	void* p = alloc.allocate(std::numeric_limits<int64_t>::max());
	if (p) {
		alloc.deallocate(p);
	}
}

TEST(AllocatorTest, ArenaAllocatorOomAndResetTracking) {
	ArenaAllocator alloc(/*arena_capacity_bytes=*/64, /*arena_base_alignment=*/64,
						AllocatorConfig{16, true});

	void* a = alloc.allocate_aligned(32, 16);
	ASSERT_NE(a, nullptr);

	// Not enough room remaining.
	void* b = alloc.allocate_aligned(40, 16);
	EXPECT_EQ(b, nullptr);

	auto s = alloc.stats();
	EXPECT_EQ(s.allocations, 1u);
	EXPECT_EQ(s.live_allocations, 1u);
	EXPECT_EQ(s.live_bytes, 32u);

	// Individual deallocate is a no-op for memory, but tracking should still update.
	EXPECT_TRUE(alloc.owns(a));
	alloc.deallocate(a);
	EXPECT_FALSE(alloc.owns(a));
	s = alloc.stats();
	EXPECT_EQ(s.frees, 1u);
	EXPECT_EQ(s.live_allocations, 0u);

	// Reset should allow reuse.
	alloc.reset();
	void* c = alloc.allocate_aligned(48, 16);
	EXPECT_NE(c, nullptr);
}

TEST(AllocatorTest, FactoryHelpersCreateAllocators) {
	auto sys = inference_engine::core::make_system_allocator(AllocatorConfig{32, true});
	ASSERT_NE(sys, nullptr);
	void* p = sys->allocate(64);
	ASSERT_NE(p, nullptr);
	sys->deallocate(p);

	auto arena = inference_engine::core::make_arena_allocator(128, 64, AllocatorConfig{16, true});
	ASSERT_NE(arena, nullptr);
	void* q = arena->allocate(16);
	EXPECT_NE(q, nullptr);
	arena->deallocate(q);
}

TEST(AllocatorTest, SystemAllocatorIsThreadSafeWithTracking) {
	SystemAllocator alloc(AllocatorConfig{alignof(std::max_align_t), true});

	constexpr int kThreads = 4;
	constexpr int kIters = 500;
	std::vector<std::thread> threads;
	threads.reserve(kThreads);

	for (int t = 0; t < kThreads; ++t) {
		threads.emplace_back([&alloc]() {
			for (int i = 0; i < kIters; ++i) {
				void* p = alloc.allocate(64);
				ASSERT_NE(p, nullptr);
				// Touch memory to ensure it's usable.
				std::memset(p, 0xAB, 64);
				alloc.deallocate(p);
			}
		});
	}

	for (auto& th : threads) {
		th.join();
	}

	const auto s = alloc.stats();
	EXPECT_EQ(s.live_allocations, 0u);
	EXPECT_EQ(s.live_bytes, 0u);
}

