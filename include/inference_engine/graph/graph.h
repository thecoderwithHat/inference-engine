#pragma once

#include "inference_engine/core/tensor.h"
#include <string>
#include <vector>
#include <memory>

namespace infer {

class Graph {
public:
    Graph();
    ~Graph();
    
    void addNode(const std::string& name);
    void addEdge(const std::string& from, const std::string& to);
    inference_engine::core::Tensor execute(const inference_engine::core::Tensor& input);
};

} // namespace infer
