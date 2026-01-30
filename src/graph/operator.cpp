
#include "inference_engine/graph/operator.h"

#include <algorithm>
#include <stdexcept>

namespace infer {

Operator::Operator(std::string op_type) : op_type_(std::move(op_type)) {
	if (op_type_.empty()) {
		throw std::invalid_argument("Operator: op_type must be non-empty");
	}
}

Operator::~Operator() = default;

void Operator::setInputs(std::vector<Value*> inputs) {
	inputs_ = std::move(inputs);
}

void Operator::setOutputs(std::vector<Value*> outputs) {
	outputs_ = std::move(outputs);
}

void Operator::addInput(Value* v) {
	inputs_.push_back(v);
}

void Operator::addOutput(Value* v) {
	outputs_.push_back(v);
}

void Operator::validate() const {
	auto is_null = [](const Value* v) { return v == nullptr; };
	if (std::any_of(inputs_.begin(), inputs_.end(), is_null)) {
		throw std::invalid_argument("Operator::validate: null input Value*");
	}
	if (std::any_of(outputs_.begin(), outputs_.end(), is_null)) {
		throw std::invalid_argument("Operator::validate: null output Value*");
	}
}

std::size_t Operator::estimateMemoryBytes() const noexcept {
	return 0;
}

} // namespace infer

