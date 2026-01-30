#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
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
class MatMulBiasOp final : public Operator {
public:
    MatMulBiasOp(int64_t in_dim, int64_t out_dim, std::vector<float> weights, std::vector<float> bias)
        : Operator("MatMulBias"), in_dim_(in_dim), out_dim_(out_dim), weights_(std::move(weights)), bias_(std::move(bias)) {
        if (weights_.size() != static_cast<std::size_t>(in_dim_ * out_dim_)) {
            throw std::invalid_argument("MatMulBiasOp: weight size mismatch");
        }
        if (bias_.size() != static_cast<std::size_t>(out_dim_)) {
            throw std::invalid_argument("MatMulBiasOp: bias size mismatch");
        }
    }

    void execute() override {
        validate();
        if (inputs().size() != 1 || outputs().size() != 1) {
            throw std::invalid_argument("MatMulBiasOp expects 1 input and 1 output");
        }

        const Value* in_val = inputs()[0];
        Value* out_val = outputs()[0];
        if (in_val == nullptr || out_val == nullptr) {
            throw std::invalid_argument("MatMulBiasOp: null Value*");
        }

        const Tensor* input = in_val->tensor();
        if (input == nullptr) {
            throw std::runtime_error("MatMulBiasOp: input tensor is null");
        }
        if (input->dtype() != DataType::FP32) {
            throw std::invalid_argument("MatMulBiasOp only supports FP32");
        }

        const auto& s = in_val->shape();
        if (s.rank() != 2 || s.dim(1) != in_dim_) {
            throw std::invalid_argument("MatMulBiasOp: expected [batch, in_dim] input shape");
        }

        const int64_t batch = s.dim(0);
        const float* x = input->data_as<float>();
        if (x == nullptr) {
            throw std::runtime_error("MatMulBiasOp: input tensor has null data");
        }

        output_buf_.assign(static_cast<std::size_t>(batch * out_dim_), 0.0f);

        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t j = 0; j < out_dim_; ++j) {
                float acc = bias_[static_cast<std::size_t>(j)];
                const std::size_t base_in = static_cast<std::size_t>(b * in_dim_);
                const std::size_t base_w = static_cast<std::size_t>(j);
                for (int64_t i = 0; i < in_dim_; ++i) {
                    acc += x[base_in + static_cast<std::size_t>(i)] *
                           weights_[static_cast<std::size_t>(i * out_dim_) + base_w];
                }
                output_buf_[static_cast<std::size_t>(b * out_dim_ + j)] = acc;
            }
        }

        output_tensor_ = Tensor(Shape({batch, out_dim_}), DataType::FP32,
                                static_cast<void*>(output_buf_.data()), false);
        out_val->setTensor(&output_tensor_);
    }

    std::unique_ptr<Operator> clone() const override {
        return std::make_unique<MatMulBiasOp>(*this);
    }

private:
    int64_t in_dim_;
    int64_t out_dim_;
    std::vector<float> weights_;
    std::vector<float> bias_;
    std::vector<float> output_buf_{};
    Tensor output_tensor_{};
};

class ReluOp final : public Operator {
public:
    ReluOp() : Operator("ReLU") {}

    void execute() override {
        validate();
        if (inputs().size() != 1 || outputs().size() != 1) {
            throw std::invalid_argument("ReLU expects 1 input and 1 output");
        }

        const Value* in_val = inputs()[0];
        Value* out_val = outputs()[0];
        if (in_val == nullptr || out_val == nullptr) {
            throw std::invalid_argument("ReLU: null Value*");
        }

        const Tensor* input = in_val->tensor();
        if (input == nullptr) {
            throw std::runtime_error("ReLU: input tensor is null");
        }
        if (input->dtype() != DataType::FP32) {
            throw std::invalid_argument("ReLU only supports FP32");
        }

        const std::size_t elems = static_cast<std::size_t>(input->num_elements());
        const float* x = input->data_as<float>();
        if (x == nullptr) {
            throw std::runtime_error("ReLU: input tensor has null data");
        }

        output_buf_.resize(elems);
        for (std::size_t idx = 0; idx < elems; ++idx) {
            output_buf_[idx] = std::max(0.0f, x[idx]);
        }

        output_tensor_ = Tensor(in_val->shape(), DataType::FP32,
                                static_cast<void*>(output_buf_.data()), false);
        out_val->setTensor(&output_tensor_);
    }

    std::unique_ptr<Operator> clone() const override {
        return std::make_unique<ReluOp>(*this);
    }

private:
    std::vector<float> output_buf_{};
    Tensor output_tensor_{};
};

class SoftmaxOp final : public Operator {
public:
    SoftmaxOp() : Operator("Softmax") {}

    void execute() override {
        validate();
        if (inputs().size() != 1 || outputs().size() != 1) {
            throw std::invalid_argument("Softmax expects 1 input and 1 output");
        }

        const Value* in_val = inputs()[0];
        Value* out_val = outputs()[0];
        if (in_val == nullptr || out_val == nullptr) {
            throw std::invalid_argument("Softmax: null Value*");
        }

        const Tensor* input = in_val->tensor();
        if (input == nullptr) {
            throw std::runtime_error("Softmax: input tensor is null");
        }
        if (input->dtype() != DataType::FP32) {
            throw std::invalid_argument("Softmax only supports FP32");
        }

        const auto& s = in_val->shape();
        if (s.rank() != 2) {
            throw std::invalid_argument("Softmax: expected 2D input [batch, classes]");
        }

        const int64_t batch = s.dim(0);
        const int64_t classes = s.dim(1);
        const float* x = input->data_as<float>();
        if (x == nullptr) {
            throw std::runtime_error("Softmax: input tensor has null data");
        }

        output_buf_.assign(static_cast<std::size_t>(batch * classes), 0.0f);

        for (int64_t b = 0; b < batch; ++b) {
            const std::size_t base = static_cast<std::size_t>(b * classes);
            float max_v = -std::numeric_limits<float>::infinity();
            for (int64_t c = 0; c < classes; ++c) {
                max_v = std::max(max_v, x[base + static_cast<std::size_t>(c)]);
            }

            float sum = 0.0f;
            for (int64_t c = 0; c < classes; ++c) {
                const float e = std::exp(x[base + static_cast<std::size_t>(c)] - max_v);
                output_buf_[base + static_cast<std::size_t>(c)] = e;
                sum += e;
            }

            const float inv_sum = (sum == 0.0f) ? 0.0f : 1.0f / sum;
            for (int64_t c = 0; c < classes; ++c) {
                output_buf_[base + static_cast<std::size_t>(c)] *= inv_sum;
            }
        }

        output_tensor_ = Tensor(s, DataType::FP32, static_cast<void*>(output_buf_.data()), false);
        out_val->setTensor(&output_tensor_);
    }

    std::unique_ptr<Operator> clone() const override {
        return std::make_unique<SoftmaxOp>(*this);
    }

private:
    std::vector<float> output_buf_{};
    Tensor output_tensor_{};
};
} // namespace

int main() {
    std::cout << "Benchmark (2-layer MLP + softmax)" << std::endl;

    constexpr int64_t batch = 16;
    constexpr int64_t in_dim = 128;
    constexpr int64_t hidden = 256;
    constexpr int64_t classes = 64;

    // Build graph
    Graph g;
    Value* x = g.createValue(Shape({batch, in_dim}), DataType::FP32, "x");
    Value* h1 = g.createValue(Shape({batch, hidden}), DataType::FP32, "h1");
    Value* h1_relu = g.createValue(Shape({batch, hidden}), DataType::FP32, "h1_relu");
    Value* logits = g.createValue(Shape({batch, classes}), DataType::FP32, "logits");
    Value* probs = g.createValue(Shape({batch, classes}), DataType::FP32, "probs");
    g.setInputs({x});
    g.setOutputs({probs});

    // Random-ish weights for repeatability.
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.2f, 0.2f);

    std::vector<float> w1(static_cast<std::size_t>(in_dim * hidden));
    std::vector<float> b1(static_cast<std::size_t>(hidden));
    for (float& v : w1) v = dist(rng);
    for (float& v : b1) v = dist(rng) * 0.1f;

    std::vector<float> w2(static_cast<std::size_t>(hidden * classes));
    std::vector<float> b2(static_cast<std::size_t>(classes));
    for (float& v : w2) v = dist(rng);
    for (float& v : b2) v = dist(rng) * 0.1f;

    Node* linear1 = g.addNode(std::make_unique<MatMulBiasOp>(in_dim, hidden, w1, b1), "linear1");
    linear1->setInputs({x});
    linear1->setOutputs({h1});

    Node* relu1 = g.addNode(std::make_unique<ReluOp>(), "relu1");
    relu1->setInputs({h1});
    relu1->setOutputs({h1_relu});

    Node* linear2 = g.addNode(std::make_unique<MatMulBiasOp>(hidden, classes, w2, b2), "linear2");
    linear2->setInputs({h1_relu});
    linear2->setOutputs({logits});

    Node* softmax = g.addNode(std::make_unique<SoftmaxOp>(), "softmax");
    softmax->setInputs({logits});
    softmax->setOutputs({probs});

    // Input tensor (external memory)
    std::vector<float> buf(static_cast<std::size_t>(batch * in_dim));
    for (std::size_t i = 0; i < buf.size(); ++i) {
        buf[i] = std::sin(static_cast<float>(i % 97) * 0.01f);
    }
    Tensor input(Shape({batch, in_dim}), DataType::FP32, static_cast<void*>(buf.data()), false);

    constexpr int warmup = 20;
    constexpr int iters = 2000;

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
