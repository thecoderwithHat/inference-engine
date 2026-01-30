
#include "inference_engine/graph/node.h"

#include "inference_engine/graph/operator.h"
#include "inference_engine/graph/value.h"

#include <algorithm>
#include <atomic>
#include <sstream>
#include <stdexcept>

namespace infer {

static std::atomic<Node::Id> g_next_node_id{1};

Node::Id Node::nextId() {
	return g_next_node_id.fetch_add(1, std::memory_order_relaxed);
}

Node::Node(Graph* graph, std::string name, std::unique_ptr<Operator> op)
	: id_(nextId()), name_(std::move(name)), graph_(graph), op_(std::move(op)) {
	if (name_.empty()) {
		name_ = "node_" + std::to_string(id_);
	}
}

Node::~Node() {
	detachFromValues();
}

void Node::setName(std::string name) {
	name_ = std::move(name);
}

void Node::setOperator(std::unique_ptr<Operator> op) {
	op_ = std::move(op);
}

void Node::resetExecutionState() noexcept {
	ready_ = false;
	scheduled_ = false;
	executed_ = false;
}

void Node::setDebugInfo(std::string info) {
	debug_info_ = std::move(info);
}

void Node::detachFromValues() {
	// Remove this node from consumer lists of all current inputs.
	for (Value* v : inputs_) {
		if (v) {
			v->removeConsumer(this);
		}
	}
	// Clear producer pointer for outputs that this node produced.
	for (Value* v : outputs_) {
		if (v && v->producer() == this) {
			v->setProducer(nullptr);
		}
	}
}

void Node::attachInputsToValues() {
	for (Value* v : inputs_) {
		if (v) {
			v->addConsumer(this);
		}
	}
}

void Node::attachOutputsToValues() {
	for (Value* v : outputs_) {
		if (v) {
			v->setProducer(this);
		}
	}
}

void Node::setInputs(std::vector<Value*> inputs) {
	// Detach from old inputs
	for (Value* v : inputs_) {
		if (v) {
			v->removeConsumer(this);
		}
	}

	inputs_ = std::move(inputs);
	attachInputsToValues();
}

void Node::setOutputs(std::vector<Value*> outputs) {
	// Detach from old outputs
	for (Value* v : outputs_) {
		if (v && v->producer() == this) {
			v->setProducer(nullptr);
		}
	}

	outputs_ = std::move(outputs);
	attachOutputsToValues();
}

void Node::addInput(Value* v) {
	inputs_.push_back(v);
	if (v) {
		v->addConsumer(this);
	}
}

void Node::addOutput(Value* v) {
	outputs_.push_back(v);
	if (v) {
		v->setProducer(this);
	}
}

std::string Node::debugString() const {
	std::ostringstream oss;
	oss << "Node{id=" << id_ << ", name=\"" << name_ << "\"";
	oss << ", op=" << (op_ ? op_->type() : std::string("null"));
	oss << ", inputs=" << inputs_.size() << ", outputs=" << outputs_.size();
	if (topo_index_.has_value()) {
		oss << ", topo=" << *topo_index_;
	}
	oss << ", ready=" << (ready_ ? "true" : "false");
	oss << ", scheduled=" << (scheduled_ ? "true" : "false");
	oss << ", executed=" << (executed_ ? "true" : "false");
	if (!debug_info_.empty()) {
		oss << ", info=\"" << debug_info_ << "\"";
	}
	oss << "}";
	return oss.str();
}

} // namespace infer

