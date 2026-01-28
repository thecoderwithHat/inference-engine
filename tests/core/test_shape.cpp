// Shape unit tests (no external test framework)

#include "inference_engine/core/shape.h"

#include <iostream>
#include <vector>

namespace {

int failures = 0;

void expect_true(bool cond, const char* expr, const char* file, int line) {
	if (!cond) {
		++failures;
		std::cerr << file << ":" << line << " EXPECT_TRUE failed: " << expr << "\n";
	}
}

void expect_eq_i64(int64_t a, int64_t b, const char* expr, const char* file, int line) {
	if (a != b) {
		++failures;
		std::cerr << file << ":" << line << " EXPECT_EQ failed: " << expr
				  << " (" << a << " != " << b << ")\n";
	}
}

#define EXPECT_TRUE(x) expect_true((x), #x, __FILE__, __LINE__)
#define EXPECT_EQ_I64(a, b) expect_eq_i64((a), (b), #a " == " #b, __FILE__, __LINE__)

using namespace inference_engine::core;

void test_basic_props() {
	Shape s({2, 3, 4});
	EXPECT_EQ_I64(s.rank(), 3);
	EXPECT_EQ_I64(s.num_elements(), 24);
	EXPECT_EQ_I64(s[0], 2);
	EXPECT_EQ_I64(s[1], 3);
	EXPECT_EQ_I64(s[2], 4);

	Shape scalar({});
	EXPECT_EQ_I64(scalar.rank(), 0);
	EXPECT_EQ_I64(scalar.num_elements(), 1);
}

void test_squeeze_unsqueeze() {
	Shape s({1, 3, 1});
	Shape squeezed = s.squeeze();
	EXPECT_EQ_I64(squeezed.rank(), 1);
	EXPECT_EQ_I64(squeezed[0], 3);

	Shape unsqueezed = squeezed.unsqueeze(0);
	EXPECT_EQ_I64(unsqueezed.rank(), 2);
	EXPECT_EQ_I64(unsqueezed[0], 1);
	EXPECT_EQ_I64(unsqueezed[1], 3);
}

void test_broadcast() {
	Shape a({2, 1, 3});
	Shape b({1, 4, 3});
	Shape out = Shape::broadcast(a, b);
	EXPECT_EQ_I64(out.rank(), 3);
	EXPECT_EQ_I64(out[0], 2);
	EXPECT_EQ_I64(out[1], 4);
	EXPECT_EQ_I64(out[2], 3);
}

void test_can_reshape() {
	Shape a({2, 3});
	Shape b({3, 2});
	Shape c({7});
	EXPECT_TRUE(Shape::can_reshape(a, b));
	EXPECT_TRUE(a.can_reshape_to(b));
	EXPECT_TRUE(Shape::can_reshape(a, Shape({6})));
	EXPECT_TRUE(!Shape::can_reshape(a, c));
}

void test_elementwise_helpers() {
	Shape s1({4});
	Shape s2({4});
	Shape out = elementwise_compatible_shape(s1, s2);
	EXPECT_EQ_I64(out.rank(), 1);
	EXPECT_EQ_I64(out[0], 4);
}

} // namespace

int main() {
	test_basic_props();
	test_squeeze_unsqueeze();
	test_broadcast();
	test_can_reshape();
	test_elementwise_helpers();

	if (failures == 0) {
		std::cout << "test_shape: OK\n";
		return 0;
	}
	std::cerr << "test_shape: FAILED (" << failures << ")\n";
	return 1;
}
