
#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace infer {

class Value;
class AttributeMap;

// Base class for all operations.
class Operator {
public:
	explicit Operator(std::string op_type);
	virtual ~Operator();

	Operator(const Operator&) = default;
	Operator& operator=(const Operator&) = default;
	Operator(Operator&&) noexcept = default;
	Operator& operator=(Operator&&) noexcept = default;

	[[nodiscard]] const std::string& type() const noexcept { return op_type_; }

	// Input and output value lists.
	[[nodiscard]] const std::vector<Value*>& inputs() const noexcept { return inputs_; }
	[[nodiscard]] const std::vector<Value*>& outputs() const noexcept { return outputs_; }
	void setInputs(std::vector<Value*> inputs);
	void setOutputs(std::vector<Value*> outputs);
	void addInput(Value* v);
	void addOutput(Value* v);

	// Attribute map reference.
	[[nodiscard]] const AttributeMap* attributes() const noexcept { return attrs_; }
	[[nodiscard]] AttributeMap* attributes() noexcept { return attrs_; }
	void setAttributes(AttributeMap* attrs) noexcept { attrs_ = attrs; }

	// Validate operator configuration (shapes/dtypes, number of inputs/outputs).
	// Default implementation checks for null inputs/outputs pointers.
	virtual void validate() const;

	// Memory requirement estimation for execution (default: 0).
	[[nodiscard]] virtual std::size_t estimateMemoryBytes() const noexcept;

	// Execute the operation. Derived ops typically read input tensors from Value::tensor()
	// and write output tensors.
	virtual void execute() = 0;

	// Clone/copy for graph optimization.
	[[nodiscard]] virtual std::unique_ptr<Operator> clone() const = 0;

protected:
	std::string op_type_;
	std::vector<Value*> inputs_{};
	std::vector<Value*> outputs_{};
	AttributeMap* attrs_{nullptr};
};

} // namespace infer

