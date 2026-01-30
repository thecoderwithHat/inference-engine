#include <iostream>
#include <vector>

#include "inference_engine/core/tensor.h"
#include "inference_engine/graph/graph.h"
#include "inference_engine/graph/node.h"
#include "inference_engine/graph/operator.h"

using inference_engine::core::DataType;
using inference_engine::core::Shape;
using inference_engine::core::Tensor;
using namespace infer;

namespace {
// A tiny demo op that forwards the input tensor to the output.
class IdentityOp final : public Operator {
public:
    IdentityOp() : Operator("Identity") {}

    void execute() override {
        validate();
        if (inputs().size() != 1 || outputs().size() != 1) {
            throw std::invalid_argument("IdentityOp expects 1 input and 1 output");
        }
        Value* in = inputs()[0];
        Value* out = outputs()[0];
        if (in == nullptr || out == nullptr) {
            throw std::invalid_argument("IdentityOp: null Value*");
        }
        if (in->tensor() == nullptr) {
            throw std::runtime_error("IdentityOp: input tensor is null (graph not bound?)");
        }
        out->setTensor(const_cast<Tensor*>(in->tensor()));
    }

    std::unique_ptr<Operator> clone() const override {
        return std::make_unique<IdentityOp>(*this);
    }
};
} // namespace

int main() {
    std::cout << "Simple Inference Example (Graph IR)" << std::endl;

    // Build a minimal graph: input -> Identity -> output
    Graph g;
    g.setModelName("toy_model");
    g.setModelVersion("0.1");

    Value* x = g.createValue(Shape({1, 3}), DataType::FP32, "x");
    Value* y = g.createValue(Shape({1, 3}), DataType::FP32, "y");
    g.setInputs({x});
    g.setOutputs({y});

    Node* n = g.addNode(std::make_unique<IdentityOp>(), "identity");
    n->setInputs({x});
    n->setOutputs({y});

    // Prepare an input tensor (wrap external memory).
    std::vector<float> buf = {1.0f, 2.0f, 3.0f};
    Tensor input(Shape({1, 3}), DataType::FP32, static_cast<void*>(buf.data()), false);

    Tensor output = g.execute(input);
    std::cout << "Output: " << output.to_string() << std::endl;
    return 0;
}
