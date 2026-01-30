
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "inference_engine/core/dtype.h"
#include "inference_engine/core/shape.h"

namespace inference_engine {
namespace core {
class Tensor;
} // namespace core
} // namespace inference_engine

namespace infer {

class Node; // forward declaration (graph IR op/node)

// Represents a value in the graph (abstract tensor reference).
// During graph construction, tensor() is typically nullptr.
// During execution, tensor() can be set to point to the realized runtime tensor.
class Value {
public:
	using Id = std::uint64_t;

	Value();
	explicit Value(std::string name);
	Value(const inference_engine::core::Shape& shape,
		  inference_engine::core::DataType dtype,
		  std::string name = "");
	Value(const inference_engine::core::Shape& shape,
		  inference_engine::core::DataType dtype,
		  const inference_engine::core::QuantizationParams& qparams,
		  std::string name = "");

	// Identity and debug info
	[[nodiscard]] Id id() const noexcept { return id_; }
	[[nodiscard]] const std::string& name() const noexcept { return name_; }
	void setName(std::string name);
	[[nodiscard]] std::string debugString() const;

	// Metadata
	[[nodiscard]] const inference_engine::core::Shape& shape() const noexcept { return shape_; }
	void setShape(inference_engine::core::Shape shape);
	[[nodiscard]] inference_engine::core::DataType dtype() const noexcept { return dtype_; }
	void setDType(inference_engine::core::DataType dtype) noexcept { dtype_ = dtype; }

	// Quantization info (optional)
	[[nodiscard]] bool hasQuantization() const noexcept { return qparams_.has_value(); }
	[[nodiscard]] const std::optional<inference_engine::core::QuantizationParams>& quantization() const noexcept {
		return qparams_;
	}
	void setQuantization(inference_engine::core::QuantizationParams qp);
	void clearQuantization() noexcept { qparams_.reset(); }

	// Graph relationships
	[[nodiscard]] Node* producer() const noexcept { return producer_; }
	void setProducer(Node* producer) noexcept { producer_ = producer; }

	[[nodiscard]] const std::vector<Node*>& consumers() const noexcept { return consumers_; }
	void addConsumer(Node* consumer);
	void removeConsumer(Node* consumer);
	[[nodiscard]] bool hasConsumer(Node* consumer) const noexcept;

	// Runtime tensor pointer (non-owning).
	[[nodiscard]] inference_engine::core::Tensor* tensor() noexcept { return tensor_; }
	[[nodiscard]] const inference_engine::core::Tensor* tensor() const noexcept { return tensor_; }
	void setTensor(inference_engine::core::Tensor* tensor) noexcept { tensor_ = tensor; }
	void clearTensor() noexcept { tensor_ = nullptr; }

private:
	static Id nextId();

	Id id_{0};
	inference_engine::core::Shape shape_{};
	inference_engine::core::DataType dtype_{inference_engine::core::DataType::UNKNOWN};
	std::string name_{};

	Node* producer_{nullptr};
	std::vector<Node*> consumers_{};

	inference_engine::core::Tensor* tensor_{nullptr};
	std::optional<inference_engine::core::QuantizationParams> qparams_{};
};

} // namespace infer

