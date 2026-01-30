
#include <gtest/gtest.h>

#include "inference_engine/graph/attributes.h"
#include "inference_engine/graph/operator.h"
#include "inference_engine/graph/value.h"

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

TEST(GraphValueTest, IdIsUnique) {
	Value a;
	Value b;
	EXPECT_NE(a.id(), b.id());
}

TEST(GraphValueTest, TracksMetadataAndTensorPointer) {
	using inference_engine::core::DataType;
	using inference_engine::core::Shape;

	Value v(Shape({1, 3, 224, 224}), DataType::FP32, "input");
	EXPECT_EQ(v.name(), "input");
	EXPECT_EQ(v.dtype(), DataType::FP32);
	EXPECT_EQ(v.shape().dims().size(), 4u);
	EXPECT_EQ(v.tensor(), nullptr);

	inference_engine::core::Tensor* fake_tensor = reinterpret_cast<inference_engine::core::Tensor*>(0x1234);
	v.setTensor(fake_tensor);
	EXPECT_EQ(v.tensor(), fake_tensor);
	v.clearTensor();
	EXPECT_EQ(v.tensor(), nullptr);
}

TEST(GraphValueTest, TracksProducerAndConsumers) {
	Value v;

	infer::Node* producer = reinterpret_cast<infer::Node*>(0x10);
	infer::Node* c1 = reinterpret_cast<infer::Node*>(0x11);
	infer::Node* c2 = reinterpret_cast<infer::Node*>(0x12);

	EXPECT_EQ(v.producer(), nullptr);
	v.setProducer(producer);
	EXPECT_EQ(v.producer(), producer);

	EXPECT_FALSE(v.hasConsumer(c1));
	v.addConsumer(c1);
	v.addConsumer(c1); // no duplicates
	v.addConsumer(c2);
	EXPECT_TRUE(v.hasConsumer(c1));
	EXPECT_TRUE(v.hasConsumer(c2));
	EXPECT_EQ(v.consumers().size(), 2u);

	v.removeConsumer(c1);
	EXPECT_FALSE(v.hasConsumer(c1));
	EXPECT_EQ(v.consumers().size(), 1u);
}

namespace {
class DummyOp final : public Operator {
public:
	DummyOp() : Operator("Dummy") {}
	void execute() override {}
	std::unique_ptr<Operator> clone() const override { return std::make_unique<DummyOp>(*this); }
};
} // namespace

TEST(GraphOperatorTest, BaseUtilitiesWork) {
	DummyOp op;
	EXPECT_EQ(op.type(), "Dummy");

	Value a;
	Value b;
	op.setInputs({&a});
	op.addOutput(&b);
	ASSERT_EQ(op.inputs().size(), 1u);
	ASSERT_EQ(op.outputs().size(), 1u);
	EXPECT_EQ(op.inputs()[0], &a);
	EXPECT_EQ(op.outputs()[0], &b);

	AttributeMap attrs;
	attrs.set("axis", 1);
	op.setAttributes(&attrs);
	ASSERT_NE(op.attributes(), nullptr);
	EXPECT_TRUE(op.attributes()->has("axis"));

	EXPECT_NO_THROW(op.validate());

	auto cloned = op.clone();
	ASSERT_NE(cloned, nullptr);
	EXPECT_EQ(cloned->type(), "Dummy");
}

int main(int argc, char** argv) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}

