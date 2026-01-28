#pragma once

#include "inference_engine/core/tensor.h"
#include <memory>
#include <string>

namespace infer {

class Graph;

class Model {
public:
    Model();
    ~Model();
    
    void load(const std::string& path);
    inference_engine::core::Tensor infer(const inference_engine::core::Tensor& input);

private:
    std::unique_ptr<Graph> graph_;
};

} // namespace infer
