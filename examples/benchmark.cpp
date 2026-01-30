#include <chrono>
#include <cstddef>
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
class IdentityOp final : public Operator {
public:
    IdentityOp() : Operator("Identity") {}

    void execute() override {
        validate();
        Value* in = inputs().empty() ? nullptr : inputs()[0];
        Value* out = outputs().empty() ? nullptr : outputs()[0];
        if (in == nullptr || out == nullptr) {
            throw std::invalid_argument("IdentityOp: missing input/output");
        }
        if (in->tensor() == nullptr) {
            throw std::runtime_error("IdentityOp: input tensor is null");
        }
        out->setTensor(const_cast<Tensor*>(in->tensor()));
    }

    std::unique_ptr<Operator> clone() const override {
        return std::make_unique<IdentityOp>(*this);
    }
};
} // namespace

int main() {
    std::cout << "Benchmark (Graph IR Identity)" << std::endl;

    // Build graph
    Graph g;
    Value* x = g.createValue(Shape({1, 1024}), DataType::FP32, "x");
    Value* y = g.createValue(Shape({1, 1024}), DataType::FP32, "y");
    g.setInputs({x});
    g.setOutputs({y});

    Node* n = g.addNode(std::make_unique<IdentityOp>(), "identity");
    n->setInputs({x});
    n->setOutputs({y});

    // Input tensor (external memory)
    std::vector<float> buf(1024);
    for (std::size_t i = 0; i < buf.size(); ++i) {
        buf[i] = static_cast<float>(i % 13);
    }
    Tensor input(Shape({1, 1024}), DataType::FP32, static_cast<void*>(buf.data()), false);

    constexpr int warmup = 100;
    constexpr int iters = 20000;

    // Warm-up
    for (int i = 0; i < warmup; ++i) {
        (void)g.execute(input);
    }

    volatile float sink = 0.0f;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        Tensor out = g.execute(input);
        sink += out.data_as<float>()[0];
    }
    const auto t1 = std::chrono::steady_clock::now();

    const auto dt_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    const double dt_s = static_cast<double>(dt_ns) * 1e-9;
    const double us_per_iter = (dt_s * 1e6) / static_cast<double>(iters);
    const double iters_per_s = static_cast<double>(iters) / dt_s;

    std::cout << "iters: " << iters << "\n";
    std::cout << "time: " << dt_s << " s\n";
    std::cout << "latency: " << us_per_iter << " us/iter\n";
    std::cout << "throughput: " << iters_per_s << " iters/s\n";
    std::cout << "sink: " << sink << "\n";
    return 0;
}
