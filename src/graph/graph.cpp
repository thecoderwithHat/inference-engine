
#include "inference_engine/graph/graph.h"

#include "inference_engine/graph/node.h"
#include "inference_engine/graph/operator.h"

#include <algorithm>
#include <queue>
#include <stdexcept>
#include <unordered_map>

namespace infer {

Graph::Graph() = default;
Graph::~Graph() = default;

void Graph::setModelName(std::string n) {
    model_name_ = std::move(n);
}

void Graph::setModelVersion(std::string v) {
    model_version_ = std::move(v);
}

Value* Graph::createValue(const inference_engine::core::Shape& shape,
                          inference_engine::core::DataType dtype,
                          std::string name) {
    values_.push_back(std::make_unique<Value>(shape, dtype, std::move(name)));
    return values_.back().get();
}

Value* Graph::createValue(const inference_engine::core::Shape& shape,
                          inference_engine::core::DataType dtype,
                          const inference_engine::core::QuantizationParams& qparams,
                          std::string name) {
    values_.push_back(std::make_unique<Value>(shape, dtype, qparams, std::move(name)));
    return values_.back().get();
}

Node* Graph::addNode(std::unique_ptr<Operator> op, std::string name) {
    nodes_.push_back(std::make_unique<Node>(this, std::move(name), std::move(op)));
    return nodes_.back().get();
}

bool Graph::removeNode(Node* node) {
    if (node == nullptr) {
        return false;
    }
    auto it = std::find_if(nodes_.begin(), nodes_.end(),
                           [node](const std::unique_ptr<Node>& n) { return n.get() == node; });
    if (it == nodes_.end()) {
        return false;
    }

    // Detach node from values explicitly before erasing.
    // (Node destructor also detaches; doing it here makes intent explicit.)
    // No direct API needed: setting empty inputs/outputs triggers detaches.
    (*it)->setInputs({});
    (*it)->setOutputs({});

    nodes_.erase(it);
    return true;
}

void Graph::setInputs(std::vector<Value*> inputs) {
    inputs_ = std::move(inputs);
}

void Graph::setOutputs(std::vector<Value*> outputs) {
    outputs_ = std::move(outputs);
}

void Graph::addInput(Value* v) {
    inputs_.push_back(v);
}

void Graph::addOutput(Value* v) {
    outputs_.push_back(v);
}

bool Graph::ownsValuePtr(const Value* v) const noexcept {
    if (v == nullptr) {
        return false;
    }
    return std::any_of(values_.begin(), values_.end(), [v](const std::unique_ptr<Value>& p) { return p.get() == v; });
}

std::vector<Node*> Graph::topologicalSort() {
    // Build in-degree based on Value producer relationships.
    std::unordered_map<Node*, std::size_t> indegree;
    indegree.reserve(nodes_.size());
    for (const auto& n : nodes_) {
        indegree[n.get()] = 0;
    }

    for (const auto& n : nodes_) {
        Node* node = n.get();
        for (Value* in : node->inputs()) {
            if (in == nullptr) continue;
            Node* p = in->producer();
            if (p != nullptr) {
                auto it = indegree.find(node);
                if (it != indegree.end()) {
                    it->second += 1;
                }
            }
        }
    }

    std::queue<Node*> q;
    for (auto& kv : indegree) {
        if (kv.second == 0) {
            q.push(kv.first);
        }
    }

    std::vector<Node*> order;
    order.reserve(nodes_.size());

    while (!q.empty()) {
        Node* n = q.front();
        q.pop();
        order.push_back(n);

        // For each outgoing edge n -> consumer_of_each_output
        for (Value* out : n->outputs()) {
            if (out == nullptr) continue;
            for (Node* consumer : out->consumers()) {
                auto it = indegree.find(consumer);
                if (it == indegree.end()) continue;
                if (it->second == 0) continue;
                it->second -= 1;
                if (it->second == 0) {
                    q.push(consumer);
                }
            }
        }
    }

    // Annotate topo indices when successful.
    if (order.size() == nodes_.size()) {
        for (std::size_t i = 0; i < order.size(); ++i) {
            order[i]->setTopoIndex(i);
        }
    } else {
        for (const auto& n : nodes_) {
            n->setTopoIndex(std::nullopt);
        }
    }

    return order;
}

void Graph::validate() const {
    // Basic structural checks
    for (const auto& n : nodes_) {
        const Node* node = n.get();
        if (node->graph() != this) {
            throw std::runtime_error("Graph::validate: node has wrong parent graph");
        }
        // Operator can be null during early construction; allow it.
        if (node->op() != nullptr) {
            node->op()->validate();
        }

        for (Value* v : node->inputs()) {
            if (v == nullptr) {
                throw std::invalid_argument("Graph::validate: node has null input Value*");
            }
            if (!ownsValuePtr(v)) {
                throw std::runtime_error("Graph::validate: node input Value* not owned by graph");
            }
            if (!v->hasConsumer(const_cast<Node*>(node))) {
                throw std::runtime_error("Graph::validate: input Value missing consumer link");
            }
        }

        for (Value* v : node->outputs()) {
            if (v == nullptr) {
                throw std::invalid_argument("Graph::validate: node has null output Value*");
            }
            if (!ownsValuePtr(v)) {
                throw std::runtime_error("Graph::validate: node output Value* not owned by graph");
            }
            if (v->producer() != node) {
                throw std::runtime_error("Graph::validate: output Value has wrong producer");
            }
        }
    }

    // Validate graph IO pointers
    for (Value* v : inputs_) {
        if (v == nullptr) throw std::invalid_argument("Graph::validate: null graph input Value*");
        if (!ownsValuePtr(v)) throw std::runtime_error("Graph::validate: graph input not owned by graph");
    }
    for (Value* v : outputs_) {
        if (v == nullptr) throw std::invalid_argument("Graph::validate: null graph output Value*");
        if (!ownsValuePtr(v)) throw std::runtime_error("Graph::validate: graph output not owned by graph");
    }

    // Cycle check via topo sort
    // (const_cast is fine here: topoIndex annotations are non-semantic)
    auto* self = const_cast<Graph*>(this);
    const auto order = self->topologicalSort();
    if (order.size() != nodes_.size()) {
        throw std::runtime_error("Graph::validate: cycle detected or dangling dependency");
    }
}

MemoryPlan Graph::planMemory() {
    MemoryPlan plan;
    const auto order = topologicalSort();
    if (order.size() != nodes_.size()) {
        // No valid plan if graph has cycles.
        return plan;
    }

    // Map node->index
    std::unordered_map<const Node*, std::size_t> node_index;
    node_index.reserve(order.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        node_index[order[i]] = i;
    }

    const std::size_t n = order.size();
    // Compute lifetimes for produced values.
    for (const auto& vptr : values_) {
        const Value* v = vptr.get();
        ValueLifetime life;

        std::size_t first = 0;
        if (v->producer() != nullptr) {
            auto it = node_index.find(v->producer());
            first = (it == node_index.end()) ? 0 : it->second;
        }
        std::size_t last = first;
        for (const Node* c : v->consumers()) {
            auto it = node_index.find(c);
            if (it != node_index.end()) {
                last = std::max(last, it->second);
            }
        }
        if (std::find(outputs_.begin(), outputs_.end(), const_cast<Value*>(v)) != outputs_.end()) {
            if (n > 0) last = std::max(last, n - 1);
        }

        // Estimate bytes based on metadata (may be 0 for UNKNOWN).
        const auto elems = v->shape().num_elements();
        const auto bpe = inference_engine::core::bytes_per_element(v->dtype());
        std::size_t bytes = 0;
        if (elems > 0 && bpe > 0) {
            const std::uint64_t prod = static_cast<std::uint64_t>(elems) * static_cast<std::uint64_t>(bpe);
            bytes = static_cast<std::size_t>(prod);
        }

        life.first_index = first;
        life.last_index = last;
        life.bytes = bytes;
        plan.lifetimes.emplace(v->id(), life);
    }

    // Compute peak via sweep over node indices.
    std::size_t peak = 0;
    for (std::size_t i = 0; i < n; ++i) {
        std::size_t live = 0;
        for (const auto& kv : plan.lifetimes) {
            const auto& lf = kv.second;
            if (lf.bytes == 0) continue;
            if (lf.first_index <= i && i <= lf.last_index) {
                live += lf.bytes;
            }
        }
        peak = std::max(peak, live);
    }
    plan.peak_bytes = peak;
    return plan;
}

void Graph::applyPass(GraphPass& pass) {
    pass.run(*this);
}

void Graph::addNode(const std::string& name) {
    // Legacy placeholder: add a node without an operator.
    (void)addNode(nullptr, name);
}

void Graph::addEdge(const std::string& from, const std::string& to) {
    // Legacy placeholder for old API.
    // This graph IR models edges through Value producer/consumer links.
    (void)from;
    (void)to;
}

inference_engine::core::Tensor Graph::execute(const inference_engine::core::Tensor& input) {
    if (nodes_.empty()) {
        return input;
    }

    // Placeholder execution: if single graph input, bind it to a runtime view.
    inference_engine::core::Tensor runtime_input = input; // shallow copy, non-owning
    if (inputs_.size() == 1 && inputs_[0] != nullptr) {
        inputs_[0]->setTensor(&runtime_input);
    }

    validate();
    const auto order = topologicalSort();
    if (order.size() != nodes_.size()) {
        throw std::runtime_error("Graph::execute: graph has cycles");
    }

    for (Node* node : order) {
        if (node == nullptr) continue;
        Operator* op = node->op();
        if (op == nullptr) {
            continue;
        }
        // Keep operator IO in sync with node IO.
        op->setInputs(node->inputs());
        op->setOutputs(node->outputs());
        op->execute();
    }

    if (outputs_.size() == 1 && outputs_[0] != nullptr && outputs_[0]->tensor() != nullptr) {
        return *(outputs_[0]->tensor());
    }

    return input;
}

} // namespace infer
