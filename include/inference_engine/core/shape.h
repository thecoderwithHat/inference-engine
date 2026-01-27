#ifndef INFERENCE_ENGINE_CORE_SHAPE_H_
#define INFERENCE_ENGINE_CORE_SHAPE_H_

/*
 * Tensor shape representation and manipulation.
 * Provides shape operations including dimension access, broadcasting,
 * reshape validation, stride calculation, and utility functions.
 */
#include <vector>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <cstddef>
#include <string>

namespace inference_engine
{
    namespace core
    {
        class Shape
        {
        public:
            Shape() = default;
            explicit Shape(const std::vector<int64_t> &dims) : dimensions_(dims) {}
            explicit Shape(std::vector<int64_t> &&dims) : dimensions_(std::move(dims)) {}
            Shape(std::initializer_list<int64_t> init) : dimensions_(init) {}

            template <typename Iter>
            Shape(Iter begin, Iter end) : dimensions_(begin, end) {}

            // copying and moving semantics

            Shape(const Shape &) = default;
            Shape(Shape &&) noexcept = default;
            Shape &operator=(const Shape &) = default;
            Shape &operator=(Shape &&) noexcept = default;

            // dimension access
            inline int64_t operator[](size_t idx) const noexcept
            {
                return dimensions_[idx];
            }
            inline int64_t at(std::size_t index) const
            {
                if (index >= dimensions_.size())
                {
                    throw std::out_of_range("Shape dimension index out of range");
                }
                return dimensions_[index];
            }

            inline int64_t dim(std::size_t index) const noexcept
            {
                return dimensions_[index];
            }

            // Shape properties
            inline std::size_t rank() const noexcept
            {
                return dimensions_.size();
            }

            inline int64_t num_elements() const noexcept
            {
                if (dimensions_.empty())
                    return 1;
                return std::accumulate(dimensions_.begin(), dimensions_.end(),
                                       int64_t(1), std::multiplies<int64_t>());
            }
            inline std::size_t size() const noexcept
            {
                return dimensions_.size();
            }

            inline const std::vector<int64_t> &dims() const noexcept
            {
                return dimensions_;
            }

            inline std::vector<int64_t> &dims_mut() noexcept
            {
                return dimensions_;
            }

            // Shape comparison
            inline bool operator==(const Shape &other) const noexcept
            {
                return dimensions_ == other.dimensions_;
            }

            inline bool operator!=(const Shape &other) const noexcept
            {
                return dimensions_ != other.dimensions_;
            }
            // Shape manipulation utilities

            /*
             * Squeeze: Remove dimensions of size 1
             * If axis is specified, only squeeze that dimension if it's 1
             * Returns a new Shape with squeezed dimensions
             */
            Shape squeeze(int axis = -1) const
            {
                std::vector<int64_t> result;

                if (axis == -1)
                {
                    // Squeeze all dimensions of size 1
                    for (int64_t dim : dimensions_)
                    {
                        if (dim != 1)
                        {
                            result.push_back(dim);
                        }
                    }
                }
                else
                {
                    // Squeeze specific axis
                    if (axis < 0)
                    {
                        axis += static_cast<int>(dimensions_.size());
                    }
                    if (axis < 0 || axis >= static_cast<int>(dimensions_.size()))
                    {
                        throw std::out_of_range("Squeeze axis out of range");
                    }
                    if (dimensions_[axis] != 1)
                    {
                        throw std::invalid_argument("Can only squeeze dimensions of size 1");
                    }
                    for (std::size_t i = 0; i < dimensions_.size(); ++i)
                    {
                        if (i != static_cast<std::size_t>(axis))
                        {
                            result.push_back(dimensions_[i]);
                        }
                    }
                }
                return Shape(result);
            }
            /*
             * Unsqueeze: Add a dimension of size 1 at specified axis
             * Returns a new Shape with an added dimension
             */
            Shape unsqueeze(int axis) const
            {
                std::vector<int64_t> result = dimensions_;

                if (axis < 0)
                {
                    axis += static_cast<int>(dimensions_.size()) + 1;
                }
                if (axis < 0 || axis > static_cast<int>(dimensions_.size()))
                {
                    throw std::out_of_range("Unsqueeze axis out of range");
                }

                result.insert(result.begin() + axis, 1);
                return Shape(result);
            }

            /*
             * Reshape: Validate if reshaping to new_shape is valid
             * Two shapes are compatible for reshape if they have the same number of elements
             * Returns true if reshape is valid, false otherwise
             */
            static bool can_reshape(const Shape &from, const Shape &to) noexcept
            {
                return from.num_elements() == to.num_elements();
            }

            bool can_reshape_to(const Shape &other) const noexcept
            {
                return can_reshape(*this, other);
            }

            /*
             * Broadcasting: Check if two shapes can be broadcast together
             * and compute the output shape
             * Returns the broadcast shape, or empty shape if incompatible
             */
            static Shape broadcast(const Shape &shape1, const Shape &shape2)
            {
                const std::vector<int64_t> &dims1 = shape1.dims();
                const std::vector<int64_t> &dims2 = shape2.dims();

                std::size_t rank1 = dims1.size();
                std::size_t rank2 = dims2.size();
                std::size_t result_rank = std::max(rank1, rank2);

                std::vector<int64_t> result(result_rank);

                // Align dimensions from the right
                int offset1 = result_rank - rank1;
                int offset2 = result_rank - rank2;

                for (std::size_t i = 0; i < result_rank; ++i)
                {
                    int64_t dim1 = (i >= static_cast<std::size_t>(offset1)) ? dims1[i - offset1] : 1;
                    int64_t dim2 = (i >= static_cast<std::size_t>(offset2)) ? dims2[i - offset2] : 1;

                    if (dim1 != dim2 && dim1 != 1 && dim2 != 1)
                    {
                        throw std::invalid_argument("Shapes cannot be broadcast together");
                    }

                    result[i] = std::max(dim1, dim2);
                }

                return Shape(result);
            }

            Shape broadcast_with(const Shape &other) const
            {
                return broadcast(*this, other);
            }
            /*
             * Stride calculation: Compute row-major (C-order) strides from shape
             * For a shape [2, 3, 4], strides are [12, 4, 1]
             * Strides[i] = product of all dimensions after dimension i
             */
            std::vector<int64_t> strides() const noexcept
            {
                if (dimensions_.empty())
                {
                    return {};
                }

                std::vector<int64_t> result(dimensions_.size());
                int64_t stride = 1;

                // Calculate strides from right to left (C-order)
                for (int i = static_cast<int>(dimensions_.size()) - 1; i >= 0; --i)
                {
                    result[i] = stride;
                    stride *= dimensions_[i];
                }

                return result;
            }
            /*
             * Flatten: Convert to 1D shape
             */
            Shape flatten() const
            {
                return Shape({num_elements()});
            }

            /*
             * Flatten to 2D: Convert to shape [batch_size, -1]
             * Useful for reshaping before fully connected layers
             */
            Shape flatten_2d(int64_t batch_size) const
            {
                int64_t elements = num_elements();
                if (elements % batch_size != 0)
                {
                    throw std::invalid_argument("Cannot flatten to 2D with given batch size");
                }
                return Shape({batch_size, elements / batch_size});
            }

        private:
            std::vector<int64_t> dimensions_;
        };

    }
}
#endif // INFERENCE_ENGINE_CORE_SHAPE_H_