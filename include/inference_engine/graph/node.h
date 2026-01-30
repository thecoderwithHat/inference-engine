
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace infer {

class Graph;
class Operator;
class Value;

// Graph node wrapping an operator instance.
class Node {
public:
	using Id = std::uint64_t;

	Node(Graph* graph, std::string name, std::unique_ptr<Operator> op);
	~Node();

	Node(const Node&) = delete;
	Node& operator=(const Node&) = delete;
	Node(Node&&) noexcept = delete;
	Node& operator=(Node&&) noexcept = delete;

	[[nodiscard]] Id id() const noexcept { return id_; }
	[[nodiscard]] const std::string& name() const noexcept { return name_; }
	void setName(std::string name);

	[[nodiscard]] Graph* graph() const noexcept { return graph_; }

	// Operator instance
	[[nodiscard]] Operator* op() noexcept { return op_.get(); }
	[[nodiscard]] const Operator* op() const noexcept { return op_.get(); }
	void setOperator(std::unique_ptr<Operator> op);

	// Value references
	[[nodiscard]] const std::vector<Value*>& inputs() const noexcept { return inputs_; }
	[[nodiscard]] const std::vector<Value*>& outputs() const noexcept { return outputs_; }
	void setInputs(std::vector<Value*> inputs);
	void setOutputs(std::vector<Value*> outputs);
	void addInput(Value* v);
	void addOutput(Value* v);

	// Topological order index (set during sort)
	[[nodiscard]] std::optional<std::size_t> topoIndex() const noexcept { return topo_index_; }
	void setTopoIndex(std::optional<std::size_t> index) noexcept { topo_index_ = index; }

	// Execution state flags for scheduling
	[[nodiscard]] bool isReady() const noexcept { return ready_; }
	[[nodiscard]] bool isScheduled() const noexcept { return scheduled_; }
	[[nodiscard]] bool isExecuted() const noexcept { return executed_; }
	void setReady(bool v) noexcept { ready_ = v; }
	void setScheduled(bool v) noexcept { scheduled_ = v; }
	void setExecuted(bool v) noexcept { executed_ = v; }
	void resetExecutionState() noexcept;

	// Debug information
	[[nodiscard]] const std::string& debugInfo() const noexcept { return debug_info_; }
	void setDebugInfo(std::string info);
	[[nodiscard]] std::string debugString() const;

private:
	static Id nextId();

	void detachFromValues();
	void attachInputsToValues();
	void attachOutputsToValues();

	Id id_{0};
	std::string name_{};
	Graph* graph_{nullptr};
	std::unique_ptr<Operator> op_{};

	std::vector<Value*> inputs_{};
	std::vector<Value*> outputs_{};

	std::optional<std::size_t> topo_index_{};

	bool ready_{false};
	bool scheduled_{false};
	bool executed_{false};

	std::string debug_info_{};
};

} // namespace infer

