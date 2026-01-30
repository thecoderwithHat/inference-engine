
#include <gtest/gtest.h>

#include "inference_engine/graph/graph.h"
#include "inference_engine/graph/node.h"
#include "inference_engine/graph/operator.h"

using namespace infer;

namespace {
class NoopOp final : public Operator {
public:
	NoopOp() : Operator("Noop") {}
	void execute() override {}
	std::unique_ptr<Operator> clone() const override { return std::make_unique<NoopOp>(*this); }
};
} // namespace

TEST(GraphNodeTest, WiresProducerAndConsumers) {
	Graph g;

	auto* x = g.createValue(inference_engine::core::Shape({1}), inference_engine::core::DataType::FP32, "x");
	auto* y = g.createValue(inference_engine::core::Shape({1}), inference_engine::core::DataType::FP32, "y");

	Node* n = g.addNode(std::make_unique<NoopOp>(), "n1");
	ASSERT_NE(n, nullptr);

	n->setInputs({x});
	n->setOutputs({y});

	EXPECT_EQ(y->producer(), n);
	EXPECT_TRUE(x->hasConsumer(n));
	ASSERT_EQ(x->consumers().size(), 1u);
	EXPECT_EQ(x->consumers()[0], n);
}

TEST(GraphNodeTest, TopologicalSortAndMemoryPlanWork) {
	Graph g;
	auto* x = g.createValue(inference_engine::core::Shape({2, 2}), inference_engine::core::DataType::FP32, "x");
	auto* y = g.createValue(inference_engine::core::Shape({2, 2}), inference_engine::core::DataType::FP32, "y");
	auto* z = g.createValue(inference_engine::core::Shape({2, 2}), inference_engine::core::DataType::FP32, "z");

	Node* n1 = g.addNode(std::make_unique<NoopOp>(), "n1");
	Node* n2 = g.addNode(std::make_unique<NoopOp>(), "n2");
	n1->setInputs({x});
	n1->setOutputs({y});
	n2->setInputs({y});
	n2->setOutputs({z});

	g.setInputs({x});
	g.setOutputs({z});

	EXPECT_NO_THROW(g.validate());

	const auto order = g.topologicalSort();
	ASSERT_EQ(order.size(), 2u);
	EXPECT_EQ(order[0], n1);
	EXPECT_EQ(order[1], n2);
	EXPECT_TRUE(n1->topoIndex().has_value());
	EXPECT_TRUE(n2->topoIndex().has_value());

	const auto plan = g.planMemory();
	// FP32 2x2 => 16 bytes per value
	EXPECT_GE(plan.peak_bytes, 16u);
	EXPECT_TRUE(plan.lifetimes.find(x->id()) != plan.lifetimes.end());
	EXPECT_TRUE(plan.lifetimes.find(y->id()) != plan.lifetimes.end());
	EXPECT_TRUE(plan.lifetimes.find(z->id()) != plan.lifetimes.end());
}

TEST(GraphNodeTest, ValidateDetectsCycles) {
	Graph g;
	auto* a = g.createValue(inference_engine::core::Shape({1}), inference_engine::core::DataType::FP32, "a");
	auto* b = g.createValue(inference_engine::core::Shape({1}), inference_engine::core::DataType::FP32, "b");

	Node* n1 = g.addNode(std::make_unique<NoopOp>(), "n1");
	Node* n2 = g.addNode(std::make_unique<NoopOp>(), "n2");
	n1->setInputs({b});
	n1->setOutputs({a});
	n2->setInputs({a});
	n2->setOutputs({b});

	EXPECT_THROW(g.validate(), std::runtime_error);
	const auto order = g.topologicalSort();
	EXPECT_NE(order.size(), 2u);
}

