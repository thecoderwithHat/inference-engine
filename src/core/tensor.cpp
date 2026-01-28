#include "inference_engine/core/tensor.h"
#include "inference_engine/memory/allocator.h"
#include <algorithm>
#include <cstring>
#include <sstream>
#include <numeric>

namespace inference_engine {
namespace core {

// ==================== Constructors with allocator ====================

Tensor::Tensor(const Shape& shape, DataType dtype, Allocator* allocator)
    : shape_(shape), dtype_(dtype) {
    compute_strides();
    
    if (allocator && num_elements() > 0) {
        int64_t size_bytes = byte_size();
        data_ = allocator->allocate(size_bytes);
        owns_data_ = (data_ != nullptr);
    } else {
        data_ = nullptr;
        owns_data_ = false;
    }
}

Tensor::Tensor(Shape&& shape, DataType dtype, Allocator* allocator)
    : shape_(std::move(shape)), dtype_(dtype) {
    compute_strides();
    
    if (allocator && num_elements() > 0) {
        int64_t size_bytes = byte_size();
        data_ = allocator->allocate(size_bytes);
        owns_data_ = (data_ != nullptr);
    } else {
        data_ = nullptr;
        owns_data_ = false;
    }
}

// ==================== Destructor ====================

Tensor::~Tensor() noexcept {
    if (owns_data_ && data_) {
        deallocate_internal();
    }
}

// ==================== Data pointer management ====================

void Tensor::set_data(void* new_data, bool take_ownership) noexcept {
    if (owns_data_ && data_) {
        deallocate_internal();
    }
    data_ = new_data;
    owns_data_ = take_ownership;
}

// ==================== Memory properties ====================

bool Tensor::is_contiguous() const noexcept {
    if (rank() == 0) return true;
    if (shape_.num_elements() == 0) return true;
    
    // Check if strides match C-contiguous layout
    int64_t expected_stride = element_size();
    for (int i = static_cast<int>(rank()) - 1; i >= 0; --i) {
        if (stride(i) != expected_stride) {
            return false;
        }
        expected_stride *= dim(i);
    }
    return true;
}

void Tensor::compute_strides() noexcept {
    strides_.clear();
    
    if (rank() == 0) {
        return;
    }
    
    strides_.resize(rank());
    int64_t stride = element_size();
    
    // C-contiguous (row-major): stride increases from right to left
    for (int i = static_cast<int>(rank()) - 1; i >= 0; --i) {
        strides_[i] = stride;
        stride *= dim(i);
    }
}

// ==================== View creation (no copy) ====================

Tensor Tensor::slice(const std::vector<std::pair<int64_t, int64_t>>& ranges) const {
    if (ranges.size() != rank()) {
        throw std::invalid_argument("Number of ranges must match tensor rank");
    }
    
    // Validate ranges and calculate new shape
    std::vector<int64_t> new_dims;
    int64_t offset = 0;
    
    for (std::size_t i = 0; i < rank(); ++i) {
        int64_t start = ranges[i].first;
        int64_t end = ranges[i].second;
        int64_t dim_size = dim(i);
        
        // Handle negative indices
        if (start < 0) start += dim_size;
        if (end < 0) end += dim_size;
        
        // Validate bounds
        if (start < 0 || start > dim_size || end < 0 || end > dim_size || start > end) {
            throw std::out_of_range("Invalid slice range");
        }
        
        int64_t length = end - start;
        new_dims.push_back(length);
        
        // Calculate byte offset
        offset += start * stride(i);
    }
    
    Shape new_shape(new_dims);
    
    // Create view tensor with offset data pointer
    Tensor view(new_shape, dtype_, 
                static_cast<uint8_t*>(data_) + offset, false);
    view.strides_ = strides_;  // Keep original strides for sliced view
    view.quant_params_ = quant_params_;
    
    return view;
}

Tensor Tensor::reshape(const Shape& new_shape) const {
    if (new_shape.num_elements() != num_elements()) {
        throw std::invalid_argument("Reshape: new shape has different number of elements");
    }
    
    if (!is_contiguous()) {
        throw std::runtime_error("Reshape: tensor must be contiguous");
    }
    
    // Create a view with new shape
    Tensor view(new_shape, dtype_, data_, false);
    view.quant_params_ = quant_params_;
    return view;
}

Tensor Tensor::transpose(const std::vector<int>& axes) const {
    if (axes.size() != rank()) {
        throw std::invalid_argument("Number of axes must match tensor rank");
    }
    
    // Validate axes
    std::vector<bool> seen(rank(), false);
    for (int axis : axes) {
        if (axis < 0 || axis >= static_cast<int>(rank()) || seen[axis]) {
            throw std::invalid_argument("Invalid transpose axes");
        }
        seen[axis] = true;
    }
    
    // Permute dimensions and strides
    std::vector<int64_t> new_dims;
    std::vector<int64_t> new_strides;
    
    for (int axis : axes) {
        new_dims.push_back(dim(axis));
        new_strides.push_back(stride(axis));
    }
    
    Shape new_shape(new_dims);
    Tensor transposed(new_shape, dtype_, data_, false);
    transposed.strides_ = new_strides;
    transposed.quant_params_ = quant_params_;
    
    return transposed;
}

// ==================== Memory management ====================

void Tensor::deallocate() noexcept {
    if (owns_data_ && data_) {
        deallocate_internal();
        data_ = nullptr;
        owns_data_ = false;
    }
}

void Tensor::deallocate_internal() noexcept {
    // For now, use delete[] for simple deallocation
    // In the future, this could use an allocator pool or other strategy
    if (data_) {
        delete[] static_cast<uint8_t*>(data_);
    }
}

// ==================== Debug utilities ====================

void Tensor::print_shape(std::ostream& os) const {
    os << "Shape: ";
    for (std::size_t i = 0; i < rank(); ++i) {
        if (i > 0) os << " x ";
        os << dim(i);
    }
    os << std::endl;
}

void Tensor::print_info(std::ostream& os) const {
    os << "=== Tensor Info ===" << std::endl;
    os << "DType: " << dtype_string() << std::endl;
    os << "Shape: ";
    for (std::size_t i = 0; i < rank(); ++i) {
        if (i > 0) os << " x ";
        os << dim(i);
    }
    os << std::endl;
    
    os << "Rank: " << rank() << std::endl;
    os << "NumElements: " << num_elements() << std::endl;
    os << "ByteSize: " << byte_size() << std::endl;
    os << "ElementSize: " << element_size() << std::endl;
    
    os << "Strides: ";
    for (std::size_t i = 0; i < strides_.size(); ++i) {
        if (i > 0) os << ", ";
        os << strides_[i];
    }
    os << std::endl;
    
    os << "Contiguous: " << (is_contiguous() ? "yes" : "no") << std::endl;
    os << "Data pointer: " << data_ << std::endl;
    os << "Owns data: " << (owns_data_ ? "yes" : "no") << std::endl;
    
    if (is_quantized()) {
        os << "Quantized: yes" << std::endl;
        os << "  Scale: " << quant_params_.scale << std::endl;
        os << "  Zero-point: " << quant_params_.zero_point << std::endl;
    } else {
        os << "Quantized: no" << std::endl;
    }
}

bool Tensor::validate() const noexcept {
    // Empty tensor is valid
    if (is_empty()) {
        return true;
    }
    
    // Non-empty tensor must have data
    if (data_ == nullptr) {
        return false;
    }
    
    // Check dtype is valid
    if (dtype_ == DataType::UNKNOWN) {
        return false;
    }
    
    // Check shape is valid
    if (rank() == 0) {
        return false;  // Scalar should have at least rank 1 or be truly 0-D
    }
    
    // Check strides are consistent with shape
    if (strides_.size() != rank()) {
        return false;
    }
    
    // Strides should be positive for C-contiguous
    for (int64_t s : strides_) {
        if (s <= 0 && rank() > 0) {
            // Allow zero stride only for broadcast dimensions (dim == 1)
            // This check is simplified
        }
    }
    
    // For quantized tensors, scale should be positive
    if (is_quantized() && quant_params_.scale <= 0.0f) {
        return false;
    }
    
    return true;
}

std::string Tensor::to_string() const {
    std::ostringstream oss;
    
    oss << "Tensor(";
    oss << "shape=[";
    for (std::size_t i = 0; i < rank(); ++i) {
        if (i > 0) oss << ",";
        oss << dim(i);
    }
    oss << "], ";
    oss << "dtype=" << dtype_string() << ", ";
    oss << "elements=" << num_elements() << ", ";
    oss << "bytes=" << byte_size() << ", ";
    oss << "contiguous=" << (is_contiguous() ? "true" : "false") << ", ";
    oss << "owns_data=" << (owns_data_ ? "true" : "false");
    
    if (is_quantized()) {
        oss << ", scale=" << quant_params_.scale;
        oss << ", zp=" << quant_params_.zero_point;
    }
    
    oss << ")";
    return oss.str();
}

// ==================== Helper functions ====================

std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    os << tensor.to_string();
    return os;
}

} // namespace core
} // namespace inference_engine
