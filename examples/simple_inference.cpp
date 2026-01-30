#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
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
    std::cout << "Complex Inference Example (2-layer MLP + softmax)" << std::endl;

    Graph g;
    g.setModelName("mlp_demo");
    g.setModelVersion("1.0");

    Value* x = g.createValue(Shape({1, 3}), DataType::FP32, "x");
    Value* h_linear = g.createValue(Shape({1, 4}), DataType::FP32, "h_linear");
    Value* h_relu = g.createValue(Shape({1, 4}), DataType::FP32, "h_relu");
    Value* logits = g.createValue(Shape({1, 2}), DataType::FP32, "logits");
    Value* probs = g.createValue(Shape({1, 2}), DataType::FP32, "probs");
    g.setInputs({x});
    g.setOutputs({probs});

    const std::vector<float> w1 = {
        0.2f, -0.3f, 0.5f, 0.1f,
        -0.4f, 0.7f, 0.2f, -0.6f,
        0.3f, 0.8f, -0.1f, 0.4f
    }; // shape [3,4]
    const std::vector<float> b1 = {0.05f, -0.02f, 0.1f, 0.0f};

    const std::vector<float> w2 = {
        0.6f, -0.2f,
        -0.1f, 0.3f,
        0.4f, 0.7f,
        -0.5f, 0.2f
    }; // shape [4,2]
    const std::vector<float> b2 = {0.01f, -0.03f};

    Node* linear1 = g.addNode(std::make_unique<MatMulBiasOp>(3, 4, w1, b1), "linear1");
    linear1->setInputs({x});
    linear1->setOutputs({h_linear});

    Node* relu = g.addNode(std::make_unique<ReluOp>(), "relu");
    relu->setInputs({h_linear});
    relu->setOutputs({h_relu});

    Node* linear2 = g.addNode(std::make_unique<MatMulBiasOp>(4, 2, w2, b2), "linear2");
    linear2->setInputs({h_relu});
    linear2->setOutputs({logits});

    Node* softmax = g.addNode(std::make_unique<SoftmaxOp>(), "softmax");
    softmax->setInputs({logits});
    softmax->setOutputs({probs});

    std::vector<float> buf = {1.0f, 2.0f, 3.0f};
    Tensor input(Shape({1, 3}), DataType::FP32, static_cast<void*>(buf.data()), false);

    Tensor output = g.execute(input);
    const float* out = output.data_as<float>();
    if (out != nullptr) {
        std::cout << "Probabilities: [" << out[0] << ", " << out[1] << "]" << std::endl;
    } else {
        std::cout << "Output tensor has no data bound" << std::endl;
    }

    return 0;
}
