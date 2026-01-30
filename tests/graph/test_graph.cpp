
#include <gtest/gtest.h>

#include "inference_engine/graph/attributes.h"

using namespace infer;

TEST(GraphAttributesTest, SetGetHasWorks) {
	AttributeMap attrs;

	EXPECT_FALSE(attrs.has("axis"));
	attrs.set(attr_names::kAxis, AttributeMap::Int{1});
	EXPECT_TRUE(attrs.has("axis"));
	EXPECT_EQ(attrs.get<AttributeMap::Int>("axis"), 1);

	attrs.set(attr_names::kEpsilon, AttributeMap::Float{1e-5});
	EXPECT_DOUBLE_EQ(attrs.get<AttributeMap::Float>("epsilon"), 1e-5);

	attrs.set("name", "relu");
	EXPECT_EQ(attrs.get<AttributeMap::String>("name"), "relu");

	attrs.set("perm", AttributeMap::Ints{0, 2, 3, 1});
	const auto& perm = attrs.get<AttributeMap::Ints>("perm");
	ASSERT_EQ(perm.size(), 4u);
	EXPECT_EQ(perm[0], 0);
	EXPECT_EQ(perm[3], 1);
}

TEST(GraphAttributesTest, TryGetPtrAndEraseWork) {
	AttributeMap attrs;
	attrs.set("alpha", 0.25);

	const auto* alpha = attrs.tryGetPtr<AttributeMap::Float>("alpha");
	ASSERT_NE(alpha, nullptr);
	EXPECT_DOUBLE_EQ(*alpha, 0.25);

	EXPECT_EQ(attrs.tryGetPtr<AttributeMap::Int>("alpha"), nullptr);
	EXPECT_EQ(attrs.tryGetPtr<AttributeMap::Float>("missing"), nullptr);

	attrs.erase("alpha");
	EXPECT_FALSE(attrs.has("alpha"));
}

TEST(GraphAttributesTest, GetThrowsOnMissingOrTypeMismatch) {
	AttributeMap attrs;
	attrs.set("axis", 1);

	EXPECT_THROW((void)attrs.get<AttributeMap::Int>("missing"), std::out_of_range);
	EXPECT_THROW((void)attrs.get<AttributeMap::String>("axis"), std::invalid_argument);
}

TEST(GraphAttributesTest, ToStringContainsKeys) {
	AttributeMap attrs;
	attrs.set("axis", 1);
	attrs.set("name", "conv");
	const std::string s = attrs.toString();
	EXPECT_NE(s.find("\"axis\""), std::string::npos);
	EXPECT_NE(s.find("\"name\""), std::string::npos);
}

int main(int argc, char** argv) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}

