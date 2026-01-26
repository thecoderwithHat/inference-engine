#include "inference_engine/core/dtype.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace inference_engine {
namespace core {

// ==============================================================================
// Runtime Type Trait Helpers (already constexpr in header, no implementation needed)
// ==============================================================================
// is_floating_point, is_integer, is_signed, is_unsigned, is_bool, is_quantized
// are all constexpr inline in the header

// ==============================================================================
// Quantization Helper Functions
// ==============================================================================

namespace {

// Clamp value to range [min, max]
template<typename T>
inline T clamp(T value, T min_val, T max_val) {
    return std::max(min_val, std::min(value, max_val));
}

} // anonymous namespace

// Quantize float to int8 using symmetric quantization (zero_point = 0)
int8_t quantize_symmetric_int8(float value, float scale) {
    if (scale <= 0.0f) {
        throw std::invalid_argument("Quantization scale must be positive");
    }
    
    float scaled = std::round(value / scale);
    return static_cast<int8_t>(clamp(scaled, -128.0f, 127.0f));
}

// Quantize float to uint8 using asymmetric quantization
uint8_t quantize_asymmetric_uint8(float value, float scale, int32_t zero_point) {
    if (scale <= 0.0f) {
        throw std::invalid_argument("Quantization scale must be positive");
    }
    
    float scaled = std::round(value / scale) + static_cast<float>(zero_point);
    return static_cast<uint8_t>(clamp(scaled, 0.0f, 255.0f));
}

// Dequantize int8 to float using symmetric quantization
float dequantize_symmetric_int8(int8_t value, float scale) {
    return static_cast<float>(value) * scale;
}

// Dequantize uint8 to float using asymmetric quantization
float dequantize_asymmetric_uint8(uint8_t value, float scale, int32_t zero_point) {
    return (static_cast<float>(value) - static_cast<float>(zero_point)) * scale;
}

// ==============================================================================
// Quantization Parameter Calculation
// ==============================================================================

QuantizationParams calculate_symmetric_quant_params(
    float min_val, 
    float max_val,
    DataType target_dtype) {
    
    if (!is_quantized(target_dtype)) {
        throw std::invalid_argument("Target dtype must be INT8 or UINT8 for quantization");
    }
    
    QuantizationParams params;
    params.symmetric = true;
    params.zero_point = 0;
    
    // Find the maximum absolute value
    float abs_max = std::max(std::abs(min_val), std::abs(max_val));
    
    if (abs_max < 1e-8f) {
        // All values near zero, use unit scale
        params.scale = 1.0f;
        return params;
    }
    
    if (target_dtype == DataType::INT8) {
        // Map [-abs_max, abs_max] to [-127, 127] (leaving -128 for safety)
        params.scale = abs_max / 127.0f;
    } else { // UINT8
        // Map [0, abs_max] to [0, 255]
        params.scale = abs_max / 255.0f;
    }
    
    return params;
}

QuantizationParams calculate_asymmetric_quant_params(
    float min_val,
    float max_val,
    DataType target_dtype) {
    
    if (target_dtype != DataType::UINT8) {
        throw std::invalid_argument("Asymmetric quantization typically uses UINT8");
    }
    
    if (min_val >= max_val) {
        throw std::invalid_argument("min_val must be less than max_val");
    }
    
    QuantizationParams params;
    params.symmetric = false;
    
    // Map [min_val, max_val] to [0, 255]
    float range = max_val - min_val;
    
    if (range < 1e-8f) {
        // Nearly constant values
        params.scale = 1.0f;
        params.zero_point = static_cast<int32_t>(std::round(-min_val));
        return params;
    }
    
    params.scale = range / 255.0f;
    
    // Calculate zero point: the quantized value that represents 0.0f
    float initial_zero_point = -min_val / params.scale;
    params.zero_point = static_cast<int32_t>(std::round(initial_zero_point));
    
    // Clamp zero_point to valid uint8 range
    params.zero_point = clamp(params.zero_point, 0, 255);
    
    return params;
}

// Calculate per-channel quantization parameters
QuantizationParams calculate_per_channel_quant_params(
    const std::vector<float>& channel_min,
    const std::vector<float>& channel_max,
    int axis,
    bool symmetric,
    DataType target_dtype) {
    
    if (channel_min.size() != channel_max.size()) {
        throw std::invalid_argument("channel_min and channel_max must have same size");
    }
    
    if (channel_min.empty()) {
        throw std::invalid_argument("channel_min/max cannot be empty");
    }
    
    QuantizationParams params;
    params.axis = axis;
    params.symmetric = symmetric;
    
    size_t num_channels = channel_min.size();
    params.per_channel_scales.resize(num_channels);
    
    if (!symmetric) {
        params.per_channel_zero_points.resize(num_channels);
    }
    
    for (size_t i = 0; i < num_channels; ++i) {
        if (symmetric) {
            auto temp = calculate_symmetric_quant_params(
                channel_min[i], channel_max[i], target_dtype);
            params.per_channel_scales[i] = temp.scale;
        } else {
            auto temp = calculate_asymmetric_quant_params(
                channel_min[i], channel_max[i], target_dtype);
            params.per_channel_scales[i] = temp.scale;
            params.per_channel_zero_points[i] = temp.zero_point;
        }
    }
    
    return params;
}

// ==============================================================================
// Batch Quantization/Dequantization
// ==============================================================================

void quantize_buffer_symmetric_int8(
    const float* input,
    int8_t* output,
    size_t count,
    float scale) {
    
    if (scale <= 0.0f) {
        throw std::invalid_argument("Scale must be positive");
    }
    
    float inv_scale = 1.0f / scale;
    
    for (size_t i = 0; i < count; ++i) {
        float scaled = std::round(input[i] * inv_scale);
        output[i] = static_cast<int8_t>(clamp(scaled, -128.0f, 127.0f));
    }
}

void quantize_buffer_asymmetric_uint8(
    const float* input,
    uint8_t* output,
    size_t count,
    float scale,
    int32_t zero_point) {
    
    if (scale <= 0.0f) {
        throw std::invalid_argument("Scale must be positive");
    }
    
    float inv_scale = 1.0f / scale;
    float zp_float = static_cast<float>(zero_point);
    
    for (size_t i = 0; i < count; ++i) {
        float scaled = std::round(input[i] * inv_scale + zp_float);
        output[i] = static_cast<uint8_t>(clamp(scaled, 0.0f, 255.0f));
    }
}

void dequantize_buffer_symmetric_int8(
    const int8_t* input,
    float* output,
    size_t count,
    float scale) {
    
    for (size_t i = 0; i < count; ++i) {
        output[i] = static_cast<float>(input[i]) * scale;
    }
}

void dequantize_buffer_asymmetric_uint8(
    const uint8_t* input,
    float* output,
    size_t count,
    float scale,
    int32_t zero_point) {
    
    float zp_float = static_cast<float>(zero_point);
    
    for (size_t i = 0; i < count; ++i) {
        output[i] = (static_cast<float>(input[i]) - zp_float) * scale;
    }
}

// ==============================================================================
// Type Compatibility and Promotion
// ==============================================================================

bool can_cast_dtype(DataType from, DataType to) {
    // Same type always compatible
    if (from == to) {
        return true;
    }
    
    // Float to float
    if (is_floating_point(from) && is_floating_point(to)) {
        return true;
    }
    
    // Integer to integer (with potential precision change)
    if (is_integer(from) && is_integer(to)) {
        return true;
    }
    
    // Float to integer (quantization)
    if (is_floating_point(from) && is_integer(to)) {
        return true;
    }
    
    // Integer to float (dequantization)
    if (is_integer(from) && is_floating_point(to)) {
        return true;
    }
    
    // Bool can convert to any numeric type
    if (from == DataType::BOOL) {
        return to != DataType::UNKNOWN;
    }
    
    // Any numeric type can convert to bool
    if (to == DataType::BOOL && from != DataType::UNKNOWN) {
        return true;
    }
    
    return false;
}

DataType promote_dtypes(DataType a, DataType b) {
    if (a == b) {
        return a;
    }
    
    // Unknown types cannot be promoted
    if (a == DataType::UNKNOWN || b == DataType::UNKNOWN) {
        return DataType::UNKNOWN;
    }
    
    // Define precedence: FP32 > FP16 > INT64 > UINT64 > INT32 > UINT32 > 
    //                    INT16 > UINT16 > INT8 > UINT8 > BOOL
    auto get_precedence = [](DataType dt) -> int {
        switch (dt) {
            case DataType::FP32:   return 110;
            case DataType::FP16:   return 100;
            case DataType::INT64:  return 90;
            case DataType::UINT64: return 85;
            case DataType::INT32:  return 80;
            case DataType::UINT32: return 75;
            case DataType::INT16:  return 70;
            case DataType::UINT16: return 65;
            case DataType::INT8:   return 60;
            case DataType::UINT8:  return 55;
            case DataType::BOOL:   return 10;
            default:               return 0;
        }
    };
    
    return get_precedence(a) > get_precedence(b) ? a : b;
}

// ==============================================================================
// Utility Functions
// ==============================================================================

size_t get_alignment_requirement(DataType dtype) {
    // For SIMD, we want good alignment
    // Cache line is 64 bytes, but we'll use element size for minimum
    size_t element_size = bytes_per_element(dtype);
    
    // Prefer at least 32-byte alignment for SIMD (AVX2)
    // but don't go smaller than natural alignment
    if (element_size >= 4) {
        return 32; // Good for AVX2
    } else if (element_size == 2) {
        return 16; // Reasonable for smaller types
    } else {
        return 16; // Minimum for efficient access
    }
}

bool is_dtype_valid(DataType dtype) {
    return dtype != DataType::UNKNOWN && 
           static_cast<int>(dtype) >= static_cast<int>(DataType::FP32) &&
           static_cast<int>(dtype) <= static_cast<int>(DataType::BOOL);
}

} // namespace core
} // namespace inference_engine