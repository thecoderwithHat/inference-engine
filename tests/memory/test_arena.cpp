#include <gtest/gtest.h>

#include "inference_engine/memory/arena.h"

#include <cstdint>

using inference_engine::memory::Arena;

TEST(ArenaTest, AlignmentIsEnforced) {
	Arena arena(1024);
	EXPECT_EQ(arena.capacity(), 1024u);

	void* p = arena.allocate(1, 64);
	ASSERT_NE(p, nullptr);
	EXPECT_EQ(reinterpret_cast<std::uintptr_t>(p) % 64u, 0u);
}

TEST(ArenaTest, InvalidAlignmentReturnsNull) {
	Arena arena(128);
	EXPECT_EQ(arena.allocate(8, 3), nullptr); // not power-of-two
}

TEST(ArenaTest, OomDoesNotAdvance) {
	Arena arena(64);
	void* a = arena.allocate(32, 16);
	ASSERT_NE(a, nullptr);
	std::size_t used_before = arena.used();

	void* b = arena.allocate(1000, 16);
	EXPECT_EQ(b, nullptr);
	EXPECT_EQ(arena.used(), used_before);
}

TEST(ArenaTest, ResetReusesAndResetsStats) {
	Arena arena(256);
	EXPECT_EQ(arena.used(), 0u);
	EXPECT_EQ(arena.stats().allocations, 0u);

	void* a = arena.allocate(32, 16);
	void* b = arena.allocate(32, 16);
	ASSERT_NE(a, nullptr);
	ASSERT_NE(b, nullptr);

	EXPECT_EQ(arena.stats().allocations, 2u);
	EXPECT_GE(arena.stats().peak_used_bytes, arena.used());
	EXPECT_GT(arena.used(), 0u);

	arena.reset();
	EXPECT_EQ(arena.used(), 0u);
	EXPECT_EQ(arena.stats().allocations, 0u);
	EXPECT_EQ(arena.stats().peak_used_bytes, 0u);

	void* c = arena.allocate(64, 32);
	EXPECT_NE(c, nullptr);
}
