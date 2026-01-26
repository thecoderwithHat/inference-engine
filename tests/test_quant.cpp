#include "inference_engine/core/dtype.h"

#include <cstdint>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

int failures = 0;

void expect_true(bool cond, const char* expr, const char* file, int line) {
    if (!cond) {
        ++failures;
        std::cerr << file << ":" << line << " EXPECT_TRUE failed: " << expr << "\n";
    }
}

void expect_eq_i(int a, int b, const char* expr, const char* file, int line) {
    if (a != b) {
        ++failures;
        std::cerr << file << ":" << line << " EXPECT_EQ failed: " << expr
                  << " (" << a << " != " << b << ")\n";
    }
}

void expect_near(float a, float b, float tol, const char* expr, const char* file, int line) {
    if (std::fabs(a - b) > tol) {
        ++failures;
        std::cerr << file << ":" << line << " EXPECT_NEAR failed: " << expr
                  << " (" << a << " vs " << b << ", tol=" << tol << ")\n";
    }
}

#define EXPECT_TRUE(x) expect_true((x), #x, __FILE__, __LINE__)
#define EXPECT_EQ_INT(a, b) expect_eq_i((a), (b), #a " == " #b, __FILE__, __LINE__)
#define EXPECT_NEAR(a, b, tol) expect_near((a), (b), (tol), #a " ~= " #b, __FILE__, __LINE__)

} // namespace

int main() {
    using namespace inference_engine::core;

    // Scalar helpers
    {
        EXPECT_EQ_INT(static_cast<int>(quantize_symmetric_int8(0.0f, 0.5f)), 0);
        EXPECT_EQ_INT(static_cast<int>(quantize_symmetric_int8(1.0f, 0.5f)), 2);
        EXPECT_EQ_INT(static_cast<int>(quantize_symmetric_int8(1000.0f, 0.1f)), 127);
        EXPECT_EQ_INT(static_cast<int>(quantize_symmetric_int8(-1000.0f, 0.1f)), -128);

        EXPECT_EQ_INT(static_cast<int>(quantize_asymmetric_uint8(0.0f, 0.1f, 128)), 128);
        EXPECT_EQ_INT(static_cast<int>(quantize_asymmetric_uint8(1000.0f, 0.1f, 128)), 255);
        EXPECT_EQ_INT(static_cast<int>(quantize_asymmetric_uint8(-1000.0f, 0.1f, 128)), 0);

        EXPECT_NEAR(dequantize_symmetric_int8(static_cast<int8_t>(2), 0.5f), 1.0f, 1e-6f);
        EXPECT_NEAR(dequantize_asymmetric_uint8(static_cast<uint8_t>(128), 0.1f, 128), 0.0f, 1e-6f);
    }

    // Roundtrip within half-step tolerance
    {
        float scale = 0.2f;
        for (float x : { -1.0f, -0.7f, -0.1f, 0.0f, 0.1f, 0.7f, 1.0f }) {
            int8_t q = quantize_symmetric_int8(x, scale);
            float y = dequantize_symmetric_int8(q, scale);
            EXPECT_NEAR(y, x, scale * 0.5f + 1e-6f);
        }
    }

    // Batch ops
    {
        std::vector<float> in = { -1.0f, 0.0f, 1.0f };
        std::vector<int8_t> q(in.size());
        std::vector<float> out(in.size());
        float scale = 0.5f;

        quantize_buffer_symmetric_int8(in.data(), q.data(), q.size(), scale);
        EXPECT_EQ_INT(static_cast<int>(q[0]), -2);
        EXPECT_EQ_INT(static_cast<int>(q[1]), 0);
        EXPECT_EQ_INT(static_cast<int>(q[2]), 2);

        dequantize_buffer_symmetric_int8(q.data(), out.data(), out.size(), scale);
        EXPECT_NEAR(out[0], -1.0f, 1e-6f);
        EXPECT_NEAR(out[1], 0.0f, 1e-6f);
        EXPECT_NEAR(out[2], 1.0f, 1e-6f);
    }

    // Quant param calculations
    {
        auto sym = calculate_symmetric_quant_params(-1.0f, 1.0f, DataType::INT8);
        EXPECT_TRUE(sym.symmetric);
        EXPECT_EQ_INT(sym.zero_point, 0);
        EXPECT_NEAR(sym.scale, 1.0f / 127.0f, 1e-6f);

        auto asym = calculate_asymmetric_quant_params(-1.0f, 1.0f, DataType::UINT8);
        EXPECT_TRUE(!asym.symmetric);
        EXPECT_NEAR(asym.scale, 2.0f / 255.0f, 1e-6f);
        EXPECT_TRUE(asym.zero_point == 127 || asym.zero_point == 128);

        std::vector<float> cmin = { -1.0f, -2.0f };
        std::vector<float> cmax = { 1.0f, 2.0f };
        auto pc = calculate_per_channel_quant_params(cmin, cmax, /*axis=*/0, /*symmetric=*/true, DataType::INT8);
        EXPECT_TRUE(pc.is_per_channel());
        EXPECT_EQ_INT(static_cast<int>(pc.per_channel_scales.size()), 2);
        EXPECT_NEAR(pc.per_channel_scales[0], 1.0f / 127.0f, 1e-6f);
        EXPECT_NEAR(pc.per_channel_scales[1], 2.0f / 127.0f, 1e-6f);
        EXPECT_TRUE(pc.symmetric);
        EXPECT_EQ_INT(pc.axis, 0);
    }

    if (failures == 0) {
        std::cout << "test_quant: OK\n";
        return 0;
    }
    std::cerr << "test_quant: FAILED (" << failures << ")\n";
    return 1;
}
