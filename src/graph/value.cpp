
#include "inference_engine/graph/value.h"

#include "inference_engine/core/dtype.h"

#include <algorithm>
#include <atomic>
#include <sstream>

namespace infer {

static std::atomic<Value::Id> g_next_value_id{1};

Value::Id Value::nextId() {
	return g_next_value_id.fetch_add(1, std::memory_order_relaxed);
}

Value::Value() : id_(nextId()) {}

Value::Value(std::string name) : id_(nextId()), name_(std::move(name)) {}

Value::Value(const inference_engine::core::Shape& shape,
			 inference_engine::core::DataType dtype,
			 std::string name)
	: id_(nextId()), shape_(shape), dtype_(dtype), name_(std::move(name)) {}

Value::Value(const inference_engine::core::Shape& shape,
			 inference_engine::core::DataType dtype,
			 const inference_engine::core::QuantizationParams& qparams,
			 std::string name)
	: id_(nextId()), shape_(shape), dtype_(dtype), name_(std::move(name)), qparams_(qparams) {}

void Value::setName(std::string name) {
	name_ = std::move(name);
}

void Value::setShape(inference_engine::core::Shape shape) {
	shape_ = std::move(shape);
}

void Value::setQuantization(inference_engine::core::QuantizationParams qp) {
	qparams_ = std::move(qp);
}

void Value::addConsumer(Node* consumer) {
	if (consumer == nullptr) {
		return;
	}
	if (!hasConsumer(consumer)) {
		consumers_.push_back(consumer);
	}
}

void Value::removeConsumer(Node* consumer) {
	if (consumer == nullptr || consumers_.empty()) {
		return;
	}
	consumers_.erase(std::remove(consumers_.begin(), consumers_.end(), consumer), consumers_.end());
}

bool Value::hasConsumer(Node* consumer) const noexcept {
	if (consumer == nullptr) {
		return false;
	}
	return std::find(consumers_.begin(), consumers_.end(), consumer) != consumers_.end();
}

std::string Value::debugString() const {
	std::ostringstream oss;
	oss << "Value{id=" << id_;
	if (!name_.empty()) {
		oss << ", name=\"" << name_ << "\"";
	}
	oss << ", dtype=" << inference_engine::core::data_type_to_string(dtype_);
	oss << ", shape=[";
	for (std::size_t i = 0; i < shape_.dims().size(); ++i) {
		if (i) oss << ", ";
		oss << shape_.dims()[i];
	}
	oss << "]";
	oss << ", producer=" << (producer_ ? "set" : "null");
	oss << ", consumers=" << consumers_.size();
	oss << ", tensor=" << (tensor_ ? "set" : "null");
	if (qparams_.has_value()) {
		oss << ", quant={scale=" << qparams_->scale << ", zp=" << qparams_->zero_point;
		if (qparams_->is_per_channel()) {
			oss << ", per_channel=true";
		}
		oss << "}";
	}
	oss << "}";
	return oss.str();
}

} // namespace infer

