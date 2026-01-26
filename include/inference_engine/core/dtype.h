#ifndef INFERENCE_ENGINE_CORE_DTYPE_H_
#define INFERENCE_ENGINE_CORE_DTYPE_H_

/*
 * Data type definitions for the inference engine core.
 */

#include <cstdint>
#include <cstddef>
#include <string>
#include <type_traits>
#include <vector>

namespace inference_engine {
namespace core {    
    enum class DataType:int{
        UNKNOWN = 0,
        FP32 = 1,
        FP16 = 2,
        INT8 = 3,
        INT16 = 4,
        INT32 = 5,
        INT64 = 6,
        UINT8 = 7,
        UINT16 = 8,
        UINT32 = 9,
        UINT64 = 10,
        BOOL = 11
    };

    constexpr std::size_t bytes_per_element(DataType dt) noexcept {
        switch(dt){
            case DataType::FP32:
            case DataType::INT32:
            case DataType::UINT32:
                return 4;
            case DataType::FP16:
            case DataType::INT16:
            case DataType::UINT16:
                return 2;
            case DataType::INT64:
            case DataType::UINT64:
                return 8;
            case DataType::INT8:
            case DataType::UINT8:
            case DataType::BOOL:
                return 1;
            default:
                return 0;
        }
    }

inline const char* data_type_to_string(DataType dt) noexcept {
    switch (dt) {
    case DataType::UNKNOWN: return "UNKNOWN";
    case DataType::FP32: return "FP32";
    case DataType::FP16: return "FP16";
    case DataType::INT8: return "INT8";
    case DataType::INT16: return "INT16";
    case DataType::INT32: return "INT32";
    case DataType::INT64: return "INT64";
    case DataType::UINT8: return "UINT8";
    case DataType::UINT16: return "UINT16";
    case DataType::UINT32: return "UINT32";
    case DataType::UINT64: return "UINT64";
    case DataType::BOOL: return "BOOL";
    default: return "UNKNOWN";
    }
}

/*runtime trait helpers*/
constexpr bool is_floating_point(DataType dt)noexcept {
    return dt == DataType::FP16 || dt == DataType::FP32;
}
constexpr bool is_integer(DataType dt)noexcept {
    return dt == DataType::INT8 || dt == DataType::INT16 || dt == DataType::INT32 ||
           dt == DataType::INT64 || dt == DataType::UINT8 || dt == DataType::UINT16 ||
           dt == DataType::UINT32 || dt == DataType::UINT64;
}
constexpr bool is_signed(DataType dt)noexcept {
    return dt == DataType::INT8 || dt == DataType::INT16 || dt == DataType::INT32 ||
           dt == DataType::INT64;
}
constexpr bool is_unsigned(DataType dt)noexcept {
    return dt == DataType::UINT8 || dt == DataType::UINT16 || dt == DataType::UINT32 ||
           dt == DataType::UINT64 || dt == DataType::BOOL;
}
constexpr bool is_bool(DataType dt)noexcept {
    return dt == DataType::BOOL;
}
constexpr bool is_quantized(DataType dt)noexcept {
    return (dt == DataType::INT8 || dt == DataType::UINT8);
}

    /*compile time mapping form data types value to Cpp type
    usage:*/

    template<DataType DT>struct DataTypeToCppType{};
    template<>struct DataTypeToCppType<DataType::FP32>{
        using type = float;
    };
    template<>struct DataTypeToCppType<DataType::FP16>{
        using type = uint16_t; 
    };
    template<>struct DataTypeToCppType<DataType::INT8>{
        using type = int8_t;
    };
    template<>struct DataTypeToCppType<DataType::INT16>{
        using type = int16_t;
    };
    template<>struct DataTypeToCppType<DataType::INT32>{
        using type = int32_t;
    };
    template<>struct DataTypeToCppType<DataType::INT64>{
        using type = int64_t;
    };
    template<>struct DataTypeToCppType<DataType::UINT8>{
        using type = uint8_t;
    };  
    template<>struct DataTypeToCppType<DataType::UINT16>{
        using type = uint16_t;
    };
    template<>struct DataTypeToCppType<DataType::UINT32>{
        using type = uint32_t;
    };
    template<>struct DataTypeToCppType<DataType::UINT64>{
        using type = uint64_t;
    };
    template<>struct DataTypeToCppType<DataType::BOOL>{
        using type = bool;  
    };  
    template<DataType DT>
    using DataTypeToCppTypeT = typename DataTypeToCppType<DT>::type;

    /*compiletime mapping from cpp to datatype*/
    template <typename T>
    constexpr DataType cpp_type_to_datatype() noexcept {
        if constexpr (std::is_same_v<T, float>) return DataType::FP32;
        else if constexpr (std::is_same_v<T, uint16_t>) return DataType::FP16;
        else if constexpr (std::is_same_v<T, int8_t>) return DataType::INT8;
        else if constexpr (std::is_same_v<T, int16_t>) return DataType::INT16;
        else if constexpr (std::is_same_v<T, int32_t>) return DataType::INT32;
        else if constexpr (std::is_same_v<T, int64_t>) return DataType::INT64;
        else if constexpr (std::is_same_v<T, uint8_t>) return DataType::UINT8;
        else if constexpr (std::is_same_v<T, uint32_t>) return DataType::UINT32;
        else if constexpr (std::is_same_v<T, uint64_t>) return DataType::UINT64;
        else if constexpr (std::is_same_v<T, bool>) return DataType::BOOL;
        else return DataType::UNKNOWN;
    }

/* Quantization parameters supporting per-tensor and per-channel INT8/UINT8 quantization. */
struct QuantizationParams {
    // Per-tensor scale and zero_point used when per_channel_scales is empty.
    float scale = 1.0f;
    int32_t zero_point = 0;

    // Per-channel scales (optional). If non-empty, size equals number of channels and
    // zero_point is applied per-tensor unless per_channel_zero_points is provided.
    std::vector<float> per_channel_scales;
    std::vector<int32_t> per_channel_zero_points;

    // Axis for per-channel quant (e.g., channel index)
    int axis = 1;

    // Whether scales are symmetric (zero_point assumed zero)
    bool symmetric = false;

    QuantizationParams() = default;

    bool is_per_channel() const noexcept { return !per_channel_scales.empty(); }

    bool operator==(QuantizationParams const& o) const noexcept {
        return scale == o.scale && zero_point == o.zero_point &&
               per_channel_scales == o.per_channel_scales &&
               per_channel_zero_points == o.per_channel_zero_points &&
               axis == o.axis && symmetric == o.symmetric;
    }
};

// Quantization/Dequantization helpers
int8_t quantize_symmetric_int8(float value, float scale);
uint8_t quantize_asymmetric_uint8(float value, float scale, int32_t zero_point);
float dequantize_symmetric_int8(int8_t value, float scale);
float dequantize_asymmetric_uint8(uint8_t value, float scale, int32_t zero_point);

// Quantization parameter calculation
QuantizationParams calculate_symmetric_quant_params(
    float min_val, float max_val, DataType target_dtype);
QuantizationParams calculate_asymmetric_quant_params(
    float min_val, float max_val, DataType target_dtype);
QuantizationParams calculate_per_channel_quant_params(
    const std::vector<float>& channel_min,
    const std::vector<float>& channel_max,
    int axis, bool symmetric, DataType target_dtype);

// Batch quantization operations
void quantize_buffer_symmetric_int8(
    const float* input, int8_t* output, std::size_t count, float scale);
void quantize_buffer_asymmetric_uint8(
    const float* input, uint8_t* output, std::size_t count,
    float scale, int32_t zero_point);
void dequantize_buffer_symmetric_int8(
    const int8_t* input, float* output, std::size_t count, float scale);
void dequantize_buffer_asymmetric_uint8(
    const uint8_t* input, float* output, std::size_t count,
    float scale, int32_t zero_point);

// Type compatibility
bool can_cast_dtype(DataType from, DataType to);
DataType promote_dtypes(DataType a, DataType b);

// Utility functions
std::size_t get_alignment_requirement(DataType dtype);
bool is_dtype_valid(DataType dtype);



} // namespace core
} // namespace inference_engine
#endif // INFERENCE_ENGINE_CORE_DTYPE_H_