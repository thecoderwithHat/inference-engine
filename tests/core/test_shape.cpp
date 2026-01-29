#include <gtest/gtest.h>

#include "inference_engine/core/shape.h"

using namespace inference_engine::core;

TEST(ShapeTest, BasicProperties) {
	Shape s({2, 3, 4});
	EXPECT_EQ(s.rank(), 3u);
	EXPECT_EQ(s.num_elements(), 24);
	EXPECT_EQ(s[0], 2);
	EXPECT_EQ(s[1], 3);
	EXPECT_EQ(s[2], 4);

	Shape scalar({});
	EXPECT_EQ(scalar.rank(), 0u);
	EXPECT_EQ(scalar.num_elements(), 1);
}

TEST(ShapeTest, SqueezeUnsqueeze) {
	Shape s({1, 3, 1});
	Shape squeezed = s.squeeze();
	EXPECT_EQ(squeezed.rank(), 1u);
	EXPECT_EQ(squeezed[0], 3);

	Shape unsqueezed = squeezed.unsqueeze(0);
	EXPECT_EQ(unsqueezed.rank(), 2u);
	EXPECT_EQ(unsqueezed[0], 1);
	EXPECT_EQ(unsqueezed[1], 3);
}

TEST(ShapeTest, Broadcast) {
	Shape a({2, 1, 3});
	Shape b({1, 4, 3});
	Shape out = Shape::broadcast(a, b);
	EXPECT_EQ(out.rank(), 3u);
	EXPECT_EQ(out[0], 2);
	EXPECT_EQ(out[1], 4);
	EXPECT_EQ(out[2], 3);
}

TEST(ShapeTest, CanReshape) {
	Shape a({2, 3});
	Shape b({3, 2});
	Shape c({7});
	EXPECT_TRUE(Shape::can_reshape(a, b));
	EXPECT_TRUE(a.can_reshape_to(b));
	EXPECT_TRUE(Shape::can_reshape(a, Shape({6})));
	EXPECT_FALSE(Shape::can_reshape(a, c));
}

TEST(ShapeTest, ElementwiseHelpers) {
	Shape s1({4});
	Shape s2({4});
	Shape out = elementwise_compatible_shape(s1, s2);
	EXPECT_EQ(out.rank(), 1u);
	EXPECT_EQ(out[0], 4);
}
