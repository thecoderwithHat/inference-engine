#include "inference_engine/core/model.h"
#include "inference_engine/graph/graph.h"

namespace infer {

Model::Model() : graph_(std::make_unique<Graph>()) {
}

Model::~Model() = default;

void Model::load(const std::string& path) {
    // TODO: Load model from path
}

Tensor Model::infer(const Tensor& input) {
    return graph_->execute(input);
}

} // namespace infer
