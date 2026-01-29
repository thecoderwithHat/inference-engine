#include <gtest/gtest.h>

#include "inference_engine/core/tensor.h"
#include "inference_engine/core/shape.h"
#include "inference_engine/core/dtype.h"
#include "inference_engine/memory/allocator.h"

#include <cstdint>
#include <new>

namespace {

class SimpleAllocator : public inference_engine::core::Allocator {
public:
	void* allocate(int64_t size_bytes) override {
		return static_cast<void*>(new (std::nothrow) uint8_t[static_cast<std::size_t>(size_bytes)]);
	}

	void deallocate(void* ptr) noexcept override {
		delete[] static_cast<uint8_t*>(ptr);
	}
};

} // namespace

using namespace inference_engine::core;

TEST(TensorTest, BasicCreation) {
	Tensor t(Shape({2, 3}), DataType::FP32);

	EXPECT_EQ(t.rank(), 2u);
	EXPECT_EQ(t.dim(0), 2);
	EXPECT_EQ(t.dim(1), 3);
	EXPECT_EQ(t.dtype(), DataType::FP32);
	EXPECT_EQ(t.num_elements(), 6);
	EXPECT_EQ(t.byte_size(), 6 * static_cast<int64_t>(t.element_size()));

	const auto& strides = t.strides();
	EXPECT_EQ(strides.size(), 2u);
	EXPECT_EQ(strides[1], 4);   // last dim stride = element size
	EXPECT_EQ(strides[0], 12);  // 3 * element size
	EXPECT_TRUE(t.is_contiguous());
}

TEST(TensorTest, AllocatorOwnership) {
	SimpleAllocator alloc;
	Tensor t(Shape({4, 4}), DataType::INT8, &alloc);

	EXPECT_NE(t.data(), nullptr);
	EXPECT_TRUE(t.owns_data());
	EXPECT_TRUE(t.is_contiguous());
}

TEST(TensorTest, WrapExternal) {
	int8_t buffer[6] = {0, 1, 2, 3, 4, 5};
	Tensor t(Shape({2, 3}), DataType::INT8, buffer, false);

	EXPECT_FALSE(t.owns_data());
	EXPECT_EQ(t.data(), buffer);
	EXPECT_TRUE(t.is_contiguous());
	EXPECT_EQ(t.num_elements(), 6);
}

TEST(TensorTest, SliceView) {
	int32_t buffer[6] = {0, 1, 2, 3, 4, 5};
	Tensor base(Shape({2, 3}), DataType::INT32, buffer, false);

	Tensor view = base.slice({{0, 2}, {1, 3}}); // shape -> {2,2}

	EXPECT_EQ(view.rank(), 2u);
	EXPECT_EQ(view.dim(0), 2);
	EXPECT_EQ(view.dim(1), 2);
	EXPECT_EQ(reinterpret_cast<const uint8_t*>(view.data()),
	          reinterpret_cast<const uint8_t*>(buffer) + 4); // offset one element
	EXPECT_FALSE(view.is_contiguous()); // slice shares original strides
}

TEST(TensorTest, ReshapeView) {
	float buffer[6] = {0, 1, 2, 3, 4, 5};
	Tensor base(Shape({2, 3}), DataType::FP32, buffer, false);

	Tensor reshaped = base.reshape(Shape({3, 2}));
	EXPECT_EQ(reshaped.dim(0), 3);
	EXPECT_EQ(reshaped.dim(1), 2);
	EXPECT_EQ(reshaped.data(), base.data());
	EXPECT_TRUE(reshaped.is_contiguous());
}

TEST(TensorTest, TransposeView) {
	int16_t buffer[6] = {0, 1, 2, 3, 4, 5};
	Tensor base(Shape({2, 3}), DataType::INT16, buffer, false);

	Tensor tr = base.transpose({1, 0});
	EXPECT_EQ(tr.dim(0), 3);
	EXPECT_EQ(tr.dim(1), 2);
	EXPECT_EQ(tr.data(), base.data());
	EXPECT_FALSE(tr.is_contiguous());
	EXPECT_EQ(tr.stride(0), base.stride(1));
	EXPECT_EQ(tr.stride(1), base.stride(0));
}

TEST(TensorTest, QuantParams) {
	uint8_t data = 0;
	QuantParams qp(0.5f, 128);
	Tensor qt(Shape({1}), DataType::UINT8, &data, qp, false);

	EXPECT_TRUE(qt.is_quantized());
	EXPECT_EQ(qt.quant_params().zero_point, 128);
	EXPECT_FLOAT_EQ(qt.quant_params().scale, 0.5f);

	qt.set_quant_params(0.25f, 10);
	EXPECT_FLOAT_EQ(qt.quant_params().scale, 0.25f);
	EXPECT_EQ(qt.quant_params().zero_point, 10);
}
