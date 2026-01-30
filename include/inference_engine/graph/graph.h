#pragma once


#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "inference_engine/core/tensor.h"
#include "inference_engine/graph/attributes.h"
#include "inference_engine/graph/value.h"

namespace infer {

class Node;
class Operator;

struct ValueLifetime {
    std::size_t first_index = 0;
    std::size_t last_index = 0;
    std::size_t bytes = 0;
};

struct MemoryPlan {
    std::size_t peak_bytes = 0;
    std::unordered_map<Value::Id, ValueLifetime> lifetimes;
};

class GraphPass {
public:
    virtual ~GraphPass() = default;
    virtual void run(class Graph& g) = 0;
};

class Graph {
public:
    Graph();
    ~Graph();

    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;
    Graph(Graph&&) noexcept = delete;
    Graph& operator=(Graph&&) noexcept = delete;

    // Graph-level attributes (model name, version, debug metadata)
    [[nodiscard]] const std::string& modelName() const noexcept { return model_name_; }
    void setModelName(std::string n);
    [[nodiscard]] const std::string& modelVersion() const noexcept { return model_version_; }
    void setModelVersion(std::string v);

    [[nodiscard]] AttributeMap& attributes() noexcept { return attrs_; }
    [[nodiscard]] const AttributeMap& attributes() const noexcept { return attrs_; }

    // Node/value ownership
    [[nodiscard]] const std::vector<std::unique_ptr<Node>>& nodes() const noexcept { return nodes_; }
    [[nodiscard]] const std::vector<std::unique_ptr<Value>>& values() const noexcept { return values_; }

    // Create and register values
    Value* createValue(const inference_engine::core::Shape& shape,
                       inference_engine::core::DataType dtype,
                       std::string name = "");
    Value* createValue(const inference_engine::core::Shape& shape,
                       inference_engine::core::DataType dtype,
                       const inference_engine::core::QuantizationParams& qparams,
                       std::string name = "");

    // Create and register nodes
    Node* addNode(std::unique_ptr<Operator> op, std::string name = "");
    bool removeNode(Node* node);

    // Graph inputs/outputs
    [[nodiscard]] const std::vector<Value*>& inputs() const noexcept { return inputs_; }
    [[nodiscard]] const std::vector<Value*>& outputs() const noexcept { return outputs_; }
    void setInputs(std::vector<Value*> inputs);
    void setOutputs(std::vector<Value*> outputs);
    void addInput(Value* v);
    void addOutput(Value* v);

    // Topological sort & validation
    [[nodiscard]] std::vector<Node*> topologicalSort();
    void validate() const;

    // Memory planning (analyze lifetimes)
    [[nodiscard]] MemoryPlan planMemory();

    // Optimization pass application
    void applyPass(GraphPass& pass);

    // Backwards-compatible placeholders
    void addNode(const std::string& name);
    void addEdge(const std::string& from, const std::string& to);

    // Simple execution driver (placeholder): sets graph input tensor (if single input)
    // and runs operators in topological order.
    inference_engine::core::Tensor execute(const inference_engine::core::Tensor& input);

private:
    [[nodiscard]] bool ownsValuePtr(const Value* v) const noexcept;

    std::string model_name_{};
    std::string model_version_{};
    AttributeMap attrs_{};

    std::vector<std::unique_ptr<Node>> nodes_{};
    std::vector<std::unique_ptr<Value>> values_{};

    std::vector<Value*> inputs_{};
    std::vector<Value*> outputs_{};
};

} // namespace infer
