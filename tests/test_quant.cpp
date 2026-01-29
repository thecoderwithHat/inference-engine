#include <gtest/gtest.h>

#include "inference_engine/core/dtype.h"

#include <cstdint>
#include <vector>

using namespace inference_engine::core;

TEST(QuantTest, ScalarHelpers) {
	EXPECT_EQ(static_cast<int>(quantize_symmetric_int8(0.0f, 0.5f)), 0);
	EXPECT_EQ(static_cast<int>(quantize_symmetric_int8(1.0f, 0.5f)), 2);
	EXPECT_EQ(static_cast<int>(quantize_symmetric_int8(1000.0f, 0.1f)), 127);
	EXPECT_EQ(static_cast<int>(quantize_symmetric_int8(-1000.0f, 0.1f)), -128);

	EXPECT_EQ(static_cast<int>(quantize_asymmetric_uint8(0.0f, 0.1f, 128)), 128);
	EXPECT_EQ(static_cast<int>(quantize_asymmetric_uint8(1000.0f, 0.1f, 128)), 255);
	EXPECT_EQ(static_cast<int>(quantize_asymmetric_uint8(-1000.0f, 0.1f, 128)), 0);

	EXPECT_NEAR(dequantize_symmetric_int8(static_cast<int8_t>(2), 0.5f), 1.0f, 1e-6f);
	EXPECT_NEAR(dequantize_asymmetric_uint8(static_cast<uint8_t>(128), 0.1f, 128), 0.0f, 1e-6f);
}

TEST(QuantTest, RoundtripWithinHalfStep) {
	float scale = 0.2f;
	for (float x : { -1.0f, -0.7f, -0.1f, 0.0f, 0.1f, 0.7f, 1.0f }) {
		int8_t q = quantize_symmetric_int8(x, scale);
		float y = dequantize_symmetric_int8(q, scale);
		EXPECT_NEAR(y, x, scale * 0.5f + 1e-6f);
	}
}

TEST(QuantTest, BatchOps) {
	std::vector<float> in = { -1.0f, 0.0f, 1.0f };
	std::vector<int8_t> q(in.size());
	std::vector<float> out(in.size());
	float scale = 0.5f;

	quantize_buffer_symmetric_int8(in.data(), q.data(), q.size(), scale);
	EXPECT_EQ(static_cast<int>(q[0]), -2);
	EXPECT_EQ(static_cast<int>(q[1]), 0);
	EXPECT_EQ(static_cast<int>(q[2]), 2);

	dequantize_buffer_symmetric_int8(q.data(), out.data(), out.size(), scale);
	EXPECT_NEAR(out[0], -1.0f, 1e-6f);
	EXPECT_NEAR(out[1], 0.0f, 1e-6f);
	EXPECT_NEAR(out[2], 1.0f, 1e-6f);
}

TEST(QuantTest, QuantParamCalculations) {
	auto sym = calculate_symmetric_quant_params(-1.0f, 1.0f, DataType::INT8);
	EXPECT_TRUE(sym.symmetric);
	EXPECT_EQ(sym.zero_point, 0);
	EXPECT_NEAR(sym.scale, 1.0f / 127.0f, 1e-6f);

	auto asym = calculate_asymmetric_quant_params(-1.0f, 1.0f, DataType::UINT8);
	EXPECT_FALSE(asym.symmetric);
	EXPECT_NEAR(asym.scale, 2.0f / 255.0f, 1e-6f);
	EXPECT_TRUE(asym.zero_point == 127 || asym.zero_point == 128);

	std::vector<float> cmin = { -1.0f, -2.0f };
	std::vector<float> cmax = { 1.0f, 2.0f };
	auto pc = calculate_per_channel_quant_params(cmin, cmax, /*axis=*/0, /*symmetric=*/true, DataType::INT8);
	EXPECT_TRUE(pc.is_per_channel());
	EXPECT_EQ(static_cast<int>(pc.per_channel_scales.size()), 2);
	EXPECT_NEAR(pc.per_channel_scales[0], 1.0f / 127.0f, 1e-6f);
	EXPECT_NEAR(pc.per_channel_scales[1], 2.0f / 127.0f, 1e-6f);
	EXPECT_TRUE(pc.symmetric);
	EXPECT_EQ(pc.axis, 0);
}
