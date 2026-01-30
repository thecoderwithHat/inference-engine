
#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace infer {

// Operation attribute storage (compile-time parameters).
// Supported types: int, float, string and arrays of these.
class AttributeMap {
public:
	using Int = std::int64_t;
	using Float = double;
	using String = std::string;
	using Ints = std::vector<Int>;
	using Floats = std::vector<Float>;
	using Strings = std::vector<String>;

	using Attribute = std::variant<Int, Float, String, Ints, Floats, Strings>;

	AttributeMap() = default;
	~AttributeMap() = default;

	[[nodiscard]] bool has(const std::string& key) const noexcept;
	void erase(const std::string& key);
	void clear() noexcept;
	[[nodiscard]] std::size_t size() const noexcept;
	[[nodiscard]] bool empty() const noexcept;

	// Raw access for advanced use-cases (debugging/inspection).
	[[nodiscard]] const std::unordered_map<std::string, Attribute>& raw() const noexcept;

	// Type-safe setters.
	void set(const std::string& key, Int value);
	void set(const std::string& key, Float value);
	void set(const std::string& key, const String& value);
	void set(const std::string& key, String&& value);
	void set(const std::string& key, const char* value);
	void set(const std::string& key, const Ints& value);
	void set(const std::string& key, Ints&& value);
	void set(const std::string& key, const Floats& value);
	void set(const std::string& key, Floats&& value);
	void set(const std::string& key, const Strings& value);
	void set(const std::string& key, Strings&& value);

	// Convenience overloads to avoid ambiguity with numeric literals.
	template <typename T,
			  typename std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool> &&
											!std::is_same_v<T, Int>,
										int> = 0>
	void set(const std::string& key, T value) {
		set(key, static_cast<Int>(value));
	}

	template <typename T,
			  typename std::enable_if_t<std::is_floating_point_v<T> && !std::is_same_v<T, Float>, int> = 0>
	void set(const std::string& key, T value) {
		set(key, static_cast<Float>(value));
	}

	// Convenience setter: integral -> Int, floating-point -> Float.
	template <typename T>
	void setNumeric(const std::string& key, T value);

	// Type-safe getters (throws on missing key or type mismatch).
	template <typename T>
	[[nodiscard]] const T& get(const std::string& key) const;

	template <typename T>
	[[nodiscard]] T& get(const std::string& key);

	// Non-throwing typed lookup.
	template <typename T>
	[[nodiscard]] const T* tryGetPtr(const std::string& key) const noexcept;

	template <typename T>
	[[nodiscard]] T* tryGetPtr(const std::string& key) noexcept;

	template <typename T>
	[[nodiscard]] std::optional<T> tryGetCopy(const std::string& key) const;

	// Serialization helpers for debugging.
	[[nodiscard]] std::string toString() const;
	[[nodiscard]] static std::string attributeToString(const Attribute& attr);
	[[nodiscard]] static const char* attributeTypeName(const Attribute& attr) noexcept;

private:
	std::unordered_map<std::string, Attribute> attrs_;
};

// Common attribute names.
namespace attr_names {
inline constexpr const char* kAxis = "axis";
inline constexpr const char* kAxes = "axes";
inline constexpr const char* kAlpha = "alpha";
inline constexpr const char* kBeta = "beta";
inline constexpr const char* kGamma = "gamma";
inline constexpr const char* kEpsilon = "epsilon";
inline constexpr const char* kKeepDims = "keepdims";
inline constexpr const char* kPerm = "perm";
inline constexpr const char* kTransA = "transA";
inline constexpr const char* kTransB = "transB";
inline constexpr const char* kStrides = "strides";
inline constexpr const char* kPads = "pads";
inline constexpr const char* kDilations = "dilations";
inline constexpr const char* kKernelShape = "kernel_shape";
inline constexpr const char* kGroup = "group";
} // namespace attr_names

/* -------------------- Template implementations -------------------- */

namespace detail {
template <typename T>
struct is_supported_attribute_type : std::false_type {};

template <>
struct is_supported_attribute_type<AttributeMap::Int> : std::true_type {};
template <>
struct is_supported_attribute_type<AttributeMap::Float> : std::true_type {};
template <>
struct is_supported_attribute_type<AttributeMap::String> : std::true_type {};
template <>
struct is_supported_attribute_type<AttributeMap::Ints> : std::true_type {};
template <>
struct is_supported_attribute_type<AttributeMap::Floats> : std::true_type {};
template <>
struct is_supported_attribute_type<AttributeMap::Strings> : std::true_type {};
} // namespace detail

template <typename T>
void AttributeMap::setNumeric(const std::string& key, T value) {
	if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
		set(key, static_cast<Int>(value));
	} else if constexpr (std::is_floating_point_v<T>) {
		set(key, static_cast<Float>(value));
	} else {
		static_assert(sizeof(T) == 0, "setNumeric only supports integral or floating-point types");
	}
}

template <typename T>
const T& AttributeMap::get(const std::string& key) const {
	static_assert(detail::is_supported_attribute_type<T>::value,
				  "Unsupported AttributeMap::get<T> type");
	const auto it = attrs_.find(key);
	if (it == attrs_.end()) {
		throw std::out_of_range("AttributeMap::get: missing key '" + key + "'");
	}
	const T* ptr = std::get_if<T>(&it->second);
	if (ptr == nullptr) {
		throw std::invalid_argument(
			"AttributeMap::get: type mismatch for key '" + key + "' (stored=" +
			std::string(attributeTypeName(it->second)) + ")");
	}
	return *ptr;
}

template <typename T>
T& AttributeMap::get(const std::string& key) {
	static_assert(detail::is_supported_attribute_type<T>::value,
				  "Unsupported AttributeMap::get<T> type");
	auto it = attrs_.find(key);
	if (it == attrs_.end()) {
		throw std::out_of_range("AttributeMap::get: missing key '" + key + "'");
	}
	T* ptr = std::get_if<T>(&it->second);
	if (ptr == nullptr) {
		throw std::invalid_argument(
			"AttributeMap::get: type mismatch for key '" + key + "' (stored=" +
			std::string(attributeTypeName(it->second)) + ")");
	}
	return *ptr;
}

template <typename T>
const T* AttributeMap::tryGetPtr(const std::string& key) const noexcept {
	static_assert(detail::is_supported_attribute_type<T>::value,
				  "Unsupported AttributeMap::tryGetPtr<T> type");
	const auto it = attrs_.find(key);
	if (it == attrs_.end()) {
		return nullptr;
	}
	return std::get_if<T>(&it->second);
}

template <typename T>
T* AttributeMap::tryGetPtr(const std::string& key) noexcept {
	static_assert(detail::is_supported_attribute_type<T>::value,
				  "Unsupported AttributeMap::tryGetPtr<T> type");
	auto it = attrs_.find(key);
	if (it == attrs_.end()) {
		return nullptr;
	}
	return std::get_if<T>(&it->second);
}

template <typename T>
std::optional<T> AttributeMap::tryGetCopy(const std::string& key) const {
	static_assert(detail::is_supported_attribute_type<T>::value,
				  "Unsupported AttributeMap::tryGetCopy<T> type");
	const T* ptr = tryGetPtr<T>(key);
	if (ptr == nullptr) {
		return std::nullopt;
	}
	return *ptr;
}

} // namespace infer

