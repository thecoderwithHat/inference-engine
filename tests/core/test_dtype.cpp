
#include "inference_engine/core/dtype.h"

#include <cmath>
#include <iostream>

namespace {

int failures = 0;

void expect_true(bool cond, const char* expr, const char* file, int line) {
	if (!cond) {
		++failures;
		std::cerr << file << ":" << line << " EXPECT_TRUE failed: " << expr << "\n";
	}
}

void expect_eq_size(std::size_t a, std::size_t b, const char* expr, const char* file, int line) {
	if (a != b) {
		++failures;
		std::cerr << file << ":" << line << " EXPECT_EQ failed: " << expr
				  << " (" << a << " != " << b << ")\n";
	}
}

void expect_eq_int(int a, int b, const char* expr, const char* file, int line) {
	if (a != b) {
		++failures;
		std::cerr << file << ":" << line << " EXPECT_EQ failed: " << expr
				  << " (" << a << " != " << b << ")\n";
	}
}

void expect_close_float(float a, float b, float eps, const char* expr, const char* file, int line) {
	if (std::fabs(a - b) > eps) {
		++failures;
		std::cerr << file << ":" << line << " EXPECT_NEAR failed: " << expr
			  << " (" << a << " != " << b << ", eps=" << eps << ")\n";
	}
}

#define EXPECT_TRUE(x) expect_true((x), #x, __FILE__, __LINE__)
#define EXPECT_EQ_SIZE(a, b) expect_eq_size((a), (b), #a " == " #b, __FILE__, __LINE__)
#define EXPECT_EQ_INT(a, b) expect_eq_int((a), (b), #a " == " #b, __FILE__, __LINE__)
#define EXPECT_NEAR_F(a, b, eps) expect_close_float((a), (b), (eps), #a " ~= " #b, __FILE__, __LINE__)

} // namespace

int main() {
	using namespace inference_engine::core;

	// bytes_per_element
	EXPECT_EQ_SIZE(bytes_per_element(DataType::FP32), 4);
	EXPECT_EQ_SIZE(bytes_per_element(DataType::FP16), 2);
	EXPECT_EQ_SIZE(bytes_per_element(DataType::INT8), 1);
	EXPECT_EQ_SIZE(bytes_per_element(DataType::INT16), 2);
	EXPECT_EQ_SIZE(bytes_per_element(DataType::INT32), 4);
	EXPECT_EQ_SIZE(bytes_per_element(DataType::INT64), 8);
	EXPECT_EQ_SIZE(bytes_per_element(DataType::UINT8), 1);
	EXPECT_EQ_SIZE(bytes_per_element(DataType::UINT16), 2);
	EXPECT_EQ_SIZE(bytes_per_element(DataType::UINT32), 4);
	EXPECT_EQ_SIZE(bytes_per_element(DataType::UINT64), 8);
	EXPECT_EQ_SIZE(bytes_per_element(DataType::BOOL), 1);

	// to_string
	EXPECT_TRUE(std::string(data_type_to_string(DataType::FP32)) == "FP32");
	EXPECT_TRUE(std::string(data_type_to_string(DataType::UINT8)) == "UINT8");
	EXPECT_TRUE(std::string(data_type_to_string(DataType::UNKNOWN)) == "UNKNOWN");

	// Traits
	EXPECT_TRUE(is_floating_point(DataType::FP16));
	EXPECT_TRUE(is_floating_point(DataType::FP32));
	EXPECT_TRUE(!is_floating_point(DataType::INT8));

	EXPECT_TRUE(is_integer(DataType::INT32));
	EXPECT_TRUE(is_integer(DataType::UINT64));
	EXPECT_TRUE(!is_integer(DataType::FP32));
	EXPECT_TRUE(!is_integer(DataType::BOOL));

	EXPECT_TRUE(is_signed(DataType::INT8));
	EXPECT_TRUE(!is_signed(DataType::UINT8));
	EXPECT_TRUE(is_unsigned(DataType::UINT8));
	EXPECT_TRUE(is_unsigned(DataType::BOOL));

	EXPECT_TRUE(is_bool(DataType::BOOL));
	EXPECT_TRUE(!is_bool(DataType::UINT8));

	EXPECT_TRUE(is_quantized(DataType::INT8));
	EXPECT_TRUE(is_quantized(DataType::UINT8));
	EXPECT_TRUE(!is_quantized(DataType::INT16));

	// Validity + alignment
	EXPECT_TRUE(!is_dtype_valid(DataType::UNKNOWN));
	EXPECT_TRUE(is_dtype_valid(DataType::FP32));
	EXPECT_TRUE(is_dtype_valid(DataType::BOOL));

	EXPECT_EQ_SIZE(get_alignment_requirement(DataType::FP32), 32);
	EXPECT_EQ_SIZE(get_alignment_requirement(DataType::FP16), 16);
	EXPECT_EQ_SIZE(get_alignment_requirement(DataType::INT8), 16);

	// Cast rules
	EXPECT_TRUE(can_cast_dtype(DataType::FP32, DataType::INT8));
	EXPECT_TRUE(can_cast_dtype(DataType::INT8, DataType::FP32));
	EXPECT_TRUE(can_cast_dtype(DataType::BOOL, DataType::FP32));
	EXPECT_TRUE(!can_cast_dtype(DataType::UNKNOWN, DataType::FP32));

	// Promotion
	EXPECT_TRUE(promote_dtypes(DataType::FP32, DataType::INT8) == DataType::FP32);
	EXPECT_TRUE(promote_dtypes(DataType::UINT8, DataType::INT8) == DataType::INT8);
	EXPECT_TRUE(promote_dtypes(DataType::BOOL, DataType::UINT8) == DataType::UINT8);
	EXPECT_TRUE(promote_dtypes(DataType::UNKNOWN, DataType::UINT8) == DataType::UNKNOWN);

	// Quantization params structure
	QuantizationParams qp;
	EXPECT_TRUE(!qp.is_per_channel());
	qp.per_channel_scales = {0.1f, 0.2f};
	qp.per_channel_zero_points = {0, 1};
	qp.axis = 0;
	qp.symmetric = true;
	EXPECT_TRUE(qp.is_per_channel());
	QuantizationParams qp_copy = qp;
	EXPECT_TRUE(qp == qp_copy);

	// Quantize/dequantize helpers
	float scale = 0.5f;
	int8_t q = quantize_symmetric_int8(0.7f, scale); // 0.7/0.5 -> 1.4 -> round 1
	EXPECT_EQ_INT(q, 1);
	float dq = dequantize_symmetric_int8(q, scale);
	EXPECT_NEAR_F(dq, 0.5f, 1e-5f);

	if (failures == 0) {
		std::cout << "test_dtype: OK\n";
		return 0;
	}
	std::cerr << "test_dtype: FAILED (" << failures << ")\n";
	return 1;
}
