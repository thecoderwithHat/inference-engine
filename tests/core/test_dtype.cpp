#include <gtest/gtest.h>

#include "inference_engine/core/dtype.h"

#include <cmath>
#include <string>

using namespace inference_engine::core;

TEST(DTypeTest, BytesPerElement) {
	EXPECT_EQ(bytes_per_element(DataType::FP32), 4u);
	EXPECT_EQ(bytes_per_element(DataType::FP16), 2u);
	EXPECT_EQ(bytes_per_element(DataType::INT8), 1u);
	EXPECT_EQ(bytes_per_element(DataType::INT16), 2u);
	EXPECT_EQ(bytes_per_element(DataType::INT32), 4u);
	EXPECT_EQ(bytes_per_element(DataType::INT64), 8u);
	EXPECT_EQ(bytes_per_element(DataType::UINT8), 1u);
	EXPECT_EQ(bytes_per_element(DataType::UINT16), 2u);
	EXPECT_EQ(bytes_per_element(DataType::UINT32), 4u);
	EXPECT_EQ(bytes_per_element(DataType::UINT64), 8u);
	EXPECT_EQ(bytes_per_element(DataType::BOOL), 1u);
}

TEST(DTypeTest, ToString) {
	EXPECT_STREQ(data_type_to_string(DataType::FP32), "FP32");
	EXPECT_STREQ(data_type_to_string(DataType::UINT8), "UINT8");
	EXPECT_STREQ(data_type_to_string(DataType::UNKNOWN), "UNKNOWN");
}

TEST(DTypeTest, Traits) {
	EXPECT_TRUE(is_floating_point(DataType::FP16));
	EXPECT_TRUE(is_floating_point(DataType::FP32));
	EXPECT_FALSE(is_floating_point(DataType::INT8));

	EXPECT_TRUE(is_integer(DataType::INT32));
	EXPECT_TRUE(is_integer(DataType::UINT64));
	EXPECT_FALSE(is_integer(DataType::FP32));
	EXPECT_FALSE(is_integer(DataType::BOOL));

	EXPECT_TRUE(is_signed(DataType::INT8));
	EXPECT_FALSE(is_signed(DataType::UINT8));
	EXPECT_TRUE(is_unsigned(DataType::UINT8));
	EXPECT_TRUE(is_unsigned(DataType::BOOL));

	EXPECT_TRUE(is_bool(DataType::BOOL));
	EXPECT_FALSE(is_bool(DataType::UINT8));

	EXPECT_TRUE(is_quantized(DataType::INT8));
	EXPECT_TRUE(is_quantized(DataType::UINT8));
	EXPECT_FALSE(is_quantized(DataType::INT16));
}

TEST(DTypeTest, ValidityAndAlignment) {
	EXPECT_FALSE(is_dtype_valid(DataType::UNKNOWN));
	EXPECT_TRUE(is_dtype_valid(DataType::FP32));
	EXPECT_TRUE(is_dtype_valid(DataType::BOOL));

	EXPECT_EQ(get_alignment_requirement(DataType::FP32), 32u);
	EXPECT_EQ(get_alignment_requirement(DataType::FP16), 16u);
	EXPECT_EQ(get_alignment_requirement(DataType::INT8), 16u);
}

TEST(DTypeTest, CastRules) {
	EXPECT_TRUE(can_cast_dtype(DataType::FP32, DataType::INT8));
	EXPECT_TRUE(can_cast_dtype(DataType::INT8, DataType::FP32));
	EXPECT_TRUE(can_cast_dtype(DataType::BOOL, DataType::FP32));
	EXPECT_FALSE(can_cast_dtype(DataType::UNKNOWN, DataType::FP32));
}

TEST(DTypeTest, Promotion) {
	EXPECT_EQ(promote_dtypes(DataType::FP32, DataType::INT8), DataType::FP32);
	EXPECT_EQ(promote_dtypes(DataType::UINT8, DataType::INT8), DataType::INT8);
	EXPECT_EQ(promote_dtypes(DataType::BOOL, DataType::UINT8), DataType::UINT8);
	EXPECT_EQ(promote_dtypes(DataType::UNKNOWN, DataType::UINT8), DataType::UNKNOWN);
}

TEST(DTypeTest, QuantizationParams) {
	QuantizationParams qp;
	EXPECT_FALSE(qp.is_per_channel());
	qp.per_channel_scales = {0.1f, 0.2f};
	qp.per_channel_zero_points = {0, 1};
	qp.axis = 0;
	qp.symmetric = true;
	EXPECT_TRUE(qp.is_per_channel());
	QuantizationParams qp_copy = qp;
	EXPECT_EQ(qp, qp_copy);
}

TEST(DTypeTest, QuantizeDequantize) {
	float scale = 0.5f;
	int8_t q = quantize_symmetric_int8(0.7f, scale);
	EXPECT_EQ(q, 1);
	float dq = dequantize_symmetric_int8(q, scale);
	EXPECT_NEAR(dq, 0.5f, 1e-5f);
}
