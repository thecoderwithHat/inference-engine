// Tensor unit tests (no external test framework)

#include "inference_engine/core/tensor.h"
#include "inference_engine/core/shape.h"
#include "inference_engine/core/dtype.h"
#include "inference_engine/memory/allocator.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <new>
#include <vector>

namespace {

int failures = 0;

void expect_true(bool cond, const char* expr, const char* file, int line) {
	if (!cond) {
		++failures;
		std::cerr << file << ":" << line << " EXPECT_TRUE failed: " << expr << "\n";
	}
}

void expect_eq_int64(int64_t a, int64_t b, const char* expr, const char* file, int line) {
	if (a != b) {
		++failures;
		std::cerr << file << ":" << line << " EXPECT_EQ failed: " << expr
				  << " (" << a << " != " << b << ")\n";
	}
}

void expect_eq_size(std::size_t a, std::size_t b, const char* expr, const char* file, int line) {
	if (a != b) {
		++failures;
		std::cerr << file << ":" << line << " EXPECT_EQ failed: " << expr
				  << " (" << a << " != " << b << ")\n";
	}
}

void expect_eq_ptr(const void* a, const void* b, const char* expr, const char* file, int line) {
	if (a != b) {
		++failures;
		std::cerr << file << ":" << line << " EXPECT_EQ failed: " << expr
				  << " (" << a << " != " << b << ")\n";
	}
}

#define EXPECT_TRUE(x) expect_true((x), #x, __FILE__, __LINE__)
#define EXPECT_EQ_I64(a, b) expect_eq_int64((a), (b), #a " == " #b, __FILE__, __LINE__)
#define EXPECT_EQ_SIZE(a, b) expect_eq_size((a), (b), #a " == " #b, __FILE__, __LINE__)
#define EXPECT_EQ_PTR(a, b) expect_eq_ptr((a), (b), #a " == " #b, __FILE__, __LINE__)

// Simple allocator used for tests
class SimpleAllocator : public inference_engine::core::Allocator {
public:
	void* allocate(int64_t size_bytes) override {
		return static_cast<void*>(new (std::nothrow) uint8_t[static_cast<std::size_t>(size_bytes)]);
	}

	void deallocate(void* ptr) noexcept override {
		delete[] static_cast<uint8_t*>(ptr);
	}
};

using namespace inference_engine::core;

void test_basic_creation() {
	Tensor t(Shape({2, 3}), DataType::FP32);

	EXPECT_EQ_I64(t.rank(), 2);
	EXPECT_EQ_I64(t.dim(0), 2);
	EXPECT_EQ_I64(t.dim(1), 3);
	EXPECT_TRUE(t.dtype() == DataType::FP32);
	EXPECT_EQ_I64(t.num_elements(), 6);
	EXPECT_EQ_I64(t.byte_size(), 6 * static_cast<int64_t>(t.element_size()));

	const auto& strides = t.strides();
	EXPECT_EQ_SIZE(strides.size(), 2);
	EXPECT_EQ_I64(strides[1], 4);   // last dim stride = element size
	EXPECT_EQ_I64(strides[0], 12);  // 3 * element size
	EXPECT_TRUE(t.is_contiguous());
}

void test_allocator_ownership() {
	SimpleAllocator alloc;
	Tensor t(Shape({4, 4}), DataType::INT8, &alloc);

	EXPECT_TRUE(t.data() != nullptr);
	EXPECT_TRUE(t.owns_data());
	EXPECT_TRUE(t.is_contiguous());
}

void test_wrap_external() {
	int8_t buffer[6] = {0, 1, 2, 3, 4, 5};
	Tensor t(Shape({2, 3}), DataType::INT8, buffer, false);

	EXPECT_TRUE(!t.owns_data());
	EXPECT_EQ_PTR(t.data(), buffer);
	EXPECT_TRUE(t.is_contiguous());
	EXPECT_EQ_I64(t.num_elements(), 6);
}

void test_slice_view() {
	int32_t buffer[6] = {0, 1, 2, 3, 4, 5};
	Tensor base(Shape({2, 3}), DataType::INT32, buffer, false);

	Tensor view = base.slice({{0, 2}, {1, 3}}); // shape -> {2,2}

	EXPECT_EQ_I64(view.rank(), 2);
	EXPECT_EQ_I64(view.dim(0), 2);
	EXPECT_EQ_I64(view.dim(1), 2);
	EXPECT_EQ_PTR(reinterpret_cast<const uint8_t*>(view.data()),
				  reinterpret_cast<const uint8_t*>(buffer) + 4); // offset one element
	EXPECT_TRUE(!view.is_contiguous()); // slice shares original strides
}

void test_reshape_view() {
	float buffer[6] = {0, 1, 2, 3, 4, 5};
	Tensor base(Shape({2, 3}), DataType::FP32, buffer, false);

	Tensor reshaped = base.reshape(Shape({3, 2}));
	EXPECT_EQ_I64(reshaped.dim(0), 3);
	EXPECT_EQ_I64(reshaped.dim(1), 2);
	EXPECT_EQ_PTR(reshaped.data(), base.data());
	EXPECT_TRUE(reshaped.is_contiguous());
}

void test_transpose_view() {
	int16_t buffer[6] = {0, 1, 2, 3, 4, 5};
	Tensor base(Shape({2, 3}), DataType::INT16, buffer, false);

	Tensor tr = base.transpose({1, 0});
	EXPECT_EQ_I64(tr.dim(0), 3);
	EXPECT_EQ_I64(tr.dim(1), 2);
	EXPECT_EQ_PTR(tr.data(), base.data());
	EXPECT_TRUE(!tr.is_contiguous());
	EXPECT_EQ_I64(tr.stride(0), base.stride(1));
	EXPECT_EQ_I64(tr.stride(1), base.stride(0));
}

void test_quant_params() {
	uint8_t data = 0;
	QuantParams qp(0.5f, 128);
	Tensor qt(Shape({1}), DataType::UINT8, &data, qp, false);

	EXPECT_TRUE(qt.is_quantized());
	EXPECT_EQ_I64(qt.quant_params().zero_point, 128);
	EXPECT_TRUE(qt.quant_params().scale == 0.5f);

	qt.set_quant_params(0.25f, 10);
	EXPECT_TRUE(qt.quant_params().scale == 0.25f);
	EXPECT_EQ_I64(qt.quant_params().zero_point, 10);
}

} // namespace

int main() {
	test_basic_creation();
	test_allocator_ownership();
	test_wrap_external();
	test_slice_view();
	test_reshape_view();
	test_transpose_view();
	test_quant_params();

	if (failures == 0) {
		std::cout << "test_tensor: OK\n";
		return 0;
	}
	std::cerr << "test_tensor: FAILED (" << failures << ")\n";
	return 1;
}
