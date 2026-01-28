#ifndef INFERENCE_ENGINE_CORE_TENSOR_H_
#define INFERENCE_ENGINE_CORE_TENSOR_H_

/*
 * Tensor abstraction - the workhorse of the inference engine.
 * Provides a unified interface for storing and manipulating multi-dimensional arrays
 * with support for various data types, memory ownership, quantization, and view operations.
 */

#include <cstdint>
#include <cstddef>
#include <memory>
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>

#include "inference_engine/core/shape.h"
#include "inference_engine/core/dtype.h"
#include "inference_engine/core/common.h"

namespace inference_engine {
namespace core {

    /*
     * Quantization parameters for INT8/UINT8 tensors.
     * For symmetric quantization (INT8): zero_point = 0
     * For asymmetric quantization (UINT8): zero_point is typically 128 or specified value
     */
    struct QuantParams {
        float scale = 1.0f;           // Quantization scale
        int32_t zero_point = 0;       // Quantization zero point
        
        QuantParams() = default;
        QuantParams(float s, int32_t zp) : scale(s), zero_point(zp) {}
        
        bool operator==(const QuantParams& other) const noexcept {
            return scale == other.scale && zero_point == other.zero_point;
        }
    };

    class Allocator;  // Forward declaration

    /*
     * Main Tensor class representing multi-dimensional arrays.
     * Supports:
     * - Metadata (shape, dtype, strides)
     * - Typed and void pointer access to data
     * - Memory ownership tracking
     * - Quantization parameters for INT8/UINT8
     * - View creation (slicing, reshaping without copy)
     * - Contiguity checks
     * - Debug utilities
     */
    class Tensor {
    public:
        // ==================== Constructors ====================
        
        /*
         * Default constructor creates an empty tensor.
         */
        Tensor() noexcept = default;

        /*
         * Create an uninitialized tensor with specified shape and dtype.
         * Does not allocate memory - useful for specifying tensor metadata.
         */
        Tensor(const Shape& shape, DataType dtype) noexcept
            : shape_(shape), dtype_(dtype), data_(nullptr), owns_data_(false) {
            compute_strides();
        }

        Tensor(Shape&& shape, DataType dtype) noexcept
            : shape_(std::move(shape)), dtype_(dtype), data_(nullptr), owns_data_(false) {
            compute_strides();
        }

        /*
         * Create a tensor with allocated data from the provided allocator.
         * Takes ownership of the allocated memory.
         */
        Tensor(const Shape& shape, DataType dtype, Allocator* allocator);
        Tensor(Shape&& shape, DataType dtype, Allocator* allocator);

        /*
         * Create a tensor that wraps externally-managed memory.
         * does not take ownership - caller manages lifetime.
         */
        Tensor(const Shape& shape, DataType dtype, void* data, bool owns_data = false) noexcept
            : shape_(shape), dtype_(dtype), data_(data), owns_data_(owns_data) {
            compute_strides();
        }

        Tensor(Shape&& shape, DataType dtype, void* data, bool owns_data = false) noexcept
            : shape_(std::move(shape)), dtype_(dtype), data_(data), owns_data_(owns_data) {
            compute_strides();
        }

        /*
         * Create a tensor with quantization parameters (for INT8/UINT8).
         */
        Tensor(const Shape& shape, DataType dtype, void* data, 
               const QuantParams& quant_params, bool owns_data = false) noexcept
            : shape_(shape), dtype_(dtype), data_(data), owns_data_(owns_data),
              quant_params_(quant_params) {
            compute_strides();
        }

        Tensor(Shape&& shape, DataType dtype, void* data,
               const QuantParams& quant_params, bool owns_data = false) noexcept
            : shape_(std::move(shape)), dtype_(dtype), data_(data), owns_data_(owns_data),
              quant_params_(quant_params) {
            compute_strides();
        }

        // ==================== Destructor ====================
        ~Tensor() noexcept;

        // ==================== Copy semantics ====================
        
        /*
         * Shallow copy: shares data pointer, does not take ownership.
         * The copied tensor does not own the data.
         */
        Tensor(const Tensor& other) noexcept
            : shape_(other.shape_), dtype_(other.dtype_), data_(other.data_),
              owns_data_(false), strides_(other.strides_), quant_params_(other.quant_params_) {}

        Tensor& operator=(const Tensor& other) noexcept {
            if (this != &other) {
                if (owns_data_ && data_) {
                    deallocate();
                }
                shape_ = other.shape_;
                dtype_ = other.dtype_;
                data_ = other.data_;
                owns_data_ = false;
                strides_ = other.strides_;
                quant_params_ = other.quant_params_;
            }
            return *this;
        }

        // ==================== Move semantics ====================
        
        Tensor(Tensor&& other) noexcept
            : shape_(std::move(other.shape_)), dtype_(other.dtype_), 
              data_(other.data_), owns_data_(other.owns_data_), 
              strides_(std::move(other.strides_)), quant_params_(other.quant_params_) {
            other.data_ = nullptr;
            other.owns_data_ = false;
        }

        Tensor& operator=(Tensor&& other) noexcept {
            if (this != &other) {
                if (owns_data_ && data_) {
                    deallocate();
                }
                shape_ = std::move(other.shape_);
                dtype_ = other.dtype_;
                data_ = other.data_;
                owns_data_ = other.owns_data_;
                strides_ = std::move(other.strides_);
                quant_params_ = other.quant_params_;
                
                other.data_ = nullptr;
                other.owns_data_ = false;
            }
            return *this;
        }

        // ==================== Shape accessors ====================
        
        const Shape& shape() const noexcept { return shape_; }
        Shape& shape_mut() noexcept { return shape_; }
        
        int64_t dim(std::size_t index) const noexcept { return shape_.dim(index); }
        std::size_t rank() const noexcept { return shape_.rank(); }
        const std::vector<int64_t>& dims() const noexcept { return shape_.dims(); }

        // ==================== Data type accessors ====================
        
        DataType dtype() const noexcept { return dtype_; }
        
        /*
         * Get human-readable string representation of data type.
         */
        const char* dtype_string() const noexcept {
            return data_type_to_string(dtype_);
        }

        /*
         * Get size in bytes per element.
         */
        std::size_t element_size() const noexcept {
            return bytes_per_element(dtype_);
        }

        // ==================== Data pointer accessors ====================
        
        /*
         * Get void pointer to raw data (const version).
         */
        const void* data() const noexcept { return data_; }
        
        /*
         * Get void pointer to raw data (mutable version).
         */
        void* data() noexcept { return data_; }

        /*
         * Get typed pointer to data.
         * Template specializations available for all DataType enums via DataTypeToCppTypeT.
         */
        template <typename T>
        const T* data_as() const noexcept {
            return static_cast<const T*>(data_);
        }

        template <typename T>
        T* data_as() noexcept {
            return static_cast<T*>(data_);
        }

        /*
         * Set the data pointer and optionally take ownership.
         * Deallocates previous data if owned.
         */
        void set_data(void* new_data, bool take_ownership = false) noexcept;

        // ==================== Size calculations ====================
        
        /*
         * Get total number of elements in tensor.
         */
        int64_t num_elements() const noexcept { return shape_.num_elements(); }

        /*
         * Get total size in bytes (num_elements * element_size).
         */
        int64_t byte_size() const noexcept {
            return num_elements() * static_cast<int64_t>(element_size());
        }

        /*
         * Check if tensor is empty (zero-sized).
         */
        bool is_empty() const noexcept { return num_elements() == 0; }

        // ==================== Memory properties ====================
        
        /*
         * Check if this tensor owns its data.
         */
        bool owns_data() const noexcept { return owns_data_; }

        /*
         * Check if tensor data is contiguous in memory.
         */
        bool is_contiguous() const noexcept;

        /*
         * Get stride (offset in bytes) for each dimension.
         */
        const std::vector<int64_t>& strides() const noexcept { return strides_; }
        int64_t stride(std::size_t axis) const noexcept { 
            return axis < strides_.size() ? strides_[axis] : 0; 
        }

        /*
         * Compute strides from shape assuming C-contiguous (row-major) layout.
         */
        void compute_strides() noexcept;

        // ==================== Quantization ====================
        
        /*
         * Check if tensor is quantized (INT8 or UINT8).
         */
        bool is_quantized() const noexcept { return core::is_quantized(dtype_); }

        /*
         * Get quantization parameters.
         */
        const QuantParams& quant_params() const noexcept { return quant_params_; }
        QuantParams& quant_params_mut() noexcept { return quant_params_; }

        /*
         * Set quantization parameters.
         */
        void set_quant_params(const QuantParams& params) noexcept { quant_params_ = params; }
        void set_quant_params(float scale, int32_t zero_point) noexcept {
            quant_params_ = QuantParams(scale, zero_point);
        }

        // ==================== View creation (no copy) ====================
        
        /*
         * Create a slice view of this tensor without copying data.
         * Returns a new Tensor with modified shape and strides pointing to the same data.
         * Does NOT take ownership of the data.
         */
        Tensor slice(const std::vector<std::pair<int64_t, int64_t>>& ranges) const;

        /*
         * Reshape without copying data (view operation).
         * Only valid if the tensor is contiguous and the new shape has the same num_elements.
         * Returns a new Tensor with the new shape but same data pointer.
         * Does NOT take ownership.
         */
        Tensor reshape(const Shape& new_shape) const;

        /*
         * Transpose the tensor (permute dimensions).
         * For a contiguous tensor, this requires copying or at least updating strides.
         * Returns a view with modified strides for a true transpose without copy.
         */
        Tensor transpose(const std::vector<int>& axes) const;

        // ==================== Memory management ====================
        
        /*
         * Deallocate owned data. Does nothing if data is not owned.
         */
        void deallocate() noexcept;

        // ==================== Debug utilities ====================
        
        /*
         * Print tensor metadata to output stream.
         */
        void print_shape(std::ostream& os = std::cout) const;

        /*
         * Print detailed tensor information including dtype, shape, strides, etc.
         */
        void print_info(std::ostream& os = std::cout) const;

        /*
         * Validate tensor consistency:
         * - Non-null data if not empty
         * - Shape and strides consistency
         * - Valid dtype
         */
        bool validate() const noexcept;

        /*
         * Get human-readable description of tensor.
         */
        std::string to_string() const;

    private:
        Shape shape_;                          // Multi-dimensional shape
        DataType dtype_ = DataType::UNKNOWN;   // Data type
        void* data_ = nullptr;                 // Pointer to data buffer
        bool owns_data_ = false;               // Whether we manage this memory
        std::vector<int64_t> strides_;         // Strides for each dimension (in bytes)
        QuantParams quant_params_;             // Quantization parameters for INT8/UINT8

        /*
         * Deallocate data using the original allocator (if available).
         * Falls back to delete[] if allocator is not available.
         */
        void deallocate_internal() noexcept;
    };

    // ==================== Helper functions ====================
    
    /*
     * Stream output operator for debugging.
     */
    std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

    /*
     * Check if two tensors have the same shape and dtype.
     */
    inline bool shapes_match(const Tensor& t1, const Tensor& t2) noexcept {
        return t1.shape() == t2.shape() && t1.dtype() == t2.dtype();
    }

    /*
     * Check if tensor is a scalar (0-D or 1-D with 1 element).
     */
    inline bool is_scalar(const Tensor& tensor) noexcept {
        return tensor.num_elements() == 1;
    }

    /*
     * Check if tensor is a vector (1-D).
     */
    inline bool is_vector(const Tensor& tensor) noexcept {
        return tensor.rank() == 1;
    }

    /*
     * Check if tensor is a matrix (2-D).
     */
    inline bool is_matrix(const Tensor& tensor) noexcept {
        return tensor.rank() == 2;
    }

} // namespace core
} // namespace inference_engine

#endif // INFERENCE_ENGINE_CORE_TENSOR_H_
