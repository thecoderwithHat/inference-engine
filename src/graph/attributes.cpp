
#include "inference_engine/graph/attributes.h"

#include <sstream>
#include <stdexcept>

namespace infer {

bool AttributeMap::has(const std::string& key) const noexcept {
	return attrs_.find(key) != attrs_.end();
}

void AttributeMap::erase(const std::string& key) {
	attrs_.erase(key);
}

void AttributeMap::clear() noexcept {
	attrs_.clear();
}

std::size_t AttributeMap::size() const noexcept {
	return attrs_.size();
}

bool AttributeMap::empty() const noexcept {
	return attrs_.empty();
}

const std::unordered_map<std::string, AttributeMap::Attribute>& AttributeMap::raw() const noexcept {
	return attrs_;
}

void AttributeMap::set(const std::string& key, Int value) {
	attrs_[key] = value;
}

void AttributeMap::set(const std::string& key, Float value) {
	attrs_[key] = value;
}

void AttributeMap::set(const std::string& key, const String& value) {
	attrs_[key] = value;
}

void AttributeMap::set(const std::string& key, String&& value) {
	attrs_[key] = std::move(value);
}

void AttributeMap::set(const std::string& key, const char* value) {
	attrs_[key] = String(value == nullptr ? "" : value);
}

void AttributeMap::set(const std::string& key, const Ints& value) {
	attrs_[key] = value;
}

void AttributeMap::set(const std::string& key, Ints&& value) {
	attrs_[key] = std::move(value);
}

void AttributeMap::set(const std::string& key, const Floats& value) {
	attrs_[key] = value;
}

void AttributeMap::set(const std::string& key, Floats&& value) {
	attrs_[key] = std::move(value);
}

void AttributeMap::set(const std::string& key, const Strings& value) {
	attrs_[key] = value;
}

void AttributeMap::set(const std::string& key, Strings&& value) {
	attrs_[key] = std::move(value);
}

static std::string escape_string(const std::string& s) {
	std::ostringstream oss;
	for (const char c : s) {
		switch (c) {
		case '\\': oss << "\\\\"; break;
		case '"': oss << "\\\""; break;
		case '\n': oss << "\\n"; break;
		case '\r': oss << "\\r"; break;
		case '\t': oss << "\\t"; break;
		default: oss << c; break;
		}
	}
	return oss.str();
}

const char* AttributeMap::attributeTypeName(const Attribute& attr) noexcept {
	if (std::holds_alternative<Int>(attr)) return "int";
	if (std::holds_alternative<Float>(attr)) return "float";
	if (std::holds_alternative<String>(attr)) return "string";
	if (std::holds_alternative<Ints>(attr)) return "int[]";
	if (std::holds_alternative<Floats>(attr)) return "float[]";
	if (std::holds_alternative<Strings>(attr)) return "string[]";
	return "unknown";
}

std::string AttributeMap::attributeToString(const Attribute& attr) {
	std::ostringstream oss;
	std::visit(
		[&oss](const auto& v) {
			using T = std::decay_t<decltype(v)>;
			if constexpr (std::is_same_v<T, Int>) {
				oss << v;
			} else if constexpr (std::is_same_v<T, Float>) {
				oss << v;
			} else if constexpr (std::is_same_v<T, String>) {
				oss << '"' << escape_string(v) << '"';
			} else if constexpr (std::is_same_v<T, Ints> || std::is_same_v<T, Floats>) {
				oss << '[';
				for (std::size_t i = 0; i < v.size(); ++i) {
					if (i) oss << ", ";
					oss << v[i];
				}
				oss << ']';
			} else if constexpr (std::is_same_v<T, Strings>) {
				oss << '[';
				for (std::size_t i = 0; i < v.size(); ++i) {
					if (i) oss << ", ";
					oss << '"' << escape_string(v[i]) << '"';
				}
				oss << ']';
			} else {
				oss << "<unsupported>";
			}
		},
		attr);
	return oss.str();
}

std::string AttributeMap::toString() const {
	std::ostringstream oss;
	oss << "{";
	bool first = true;
	for (const auto& kv : attrs_) {
		if (!first) {
			oss << ", ";
		}
		first = false;
		oss << '"' << escape_string(kv.first) << "\": " << attributeToString(kv.second);
	}
	oss << "}";
	return oss.str();
}

} // namespace infer

