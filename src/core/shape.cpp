#include "inference_engine/core/shape.h"

namespace inference_engine {
namespace core {

// Stream output for debugging
std::ostream& operator<<(std::ostream& os, const Shape& shape) {
    os << "[";
    for (std::size_t i = 0; i < shape.rank(); ++i) {
        if (i > 0) os << ", ";
        os << shape[i];
    }
    os << "]";
    return os;
}

/*
 * Utility function to validate shape compatibility for element-wise operations
 * Returns the broadcasted output shape
 */
Shape elementwise_compatible_shape(const Shape& shape1, const Shape& shape2) {
    return Shape::broadcast(shape1, shape2);
}

/*
 * Utility function to convert a shape to a human-readable string
 */
std::string shape_to_string(const Shape& shape) {
    std::string result = "[";
    for (std::size_t i = 0; i < shape.rank(); ++i) {
        if (i > 0) result += ", ";
        result += std::to_string(shape[i]);
    }
    result += "]";
    return result;
}

/*
 * Utility function to check if a shape represents a scalar
 */
bool is_scalar(const Shape& shape) {
    return shape.rank() == 0 || (shape.rank() == 1 && shape[0] == 1);
}

/*
 * Utility function to check if a shape is a vector
 */
bool is_vector(const Shape& shape) {
    return shape.rank() == 1;
}

/*
 * Utility function to check if a shape is a matrix
 */
bool is_matrix(const Shape& shape) {
    return shape.rank() == 2;
}

/*
 * Utility function to transpose a 2D shape
 */
Shape transpose_2d(const Shape& shape) {
    if (shape.rank() != 2) {
        throw std::invalid_argument("transpose_2d requires a 2D shape");
    }
    return Shape({shape[1], shape[0]});
}

} // namespace core
} // namespace inference_engine