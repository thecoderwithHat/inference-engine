#include <gtest/gtest.h>
#include "optimizer/quantization.h"

using namespace infer;

TEST(QuantizationTest, QuantizationDequantization) {
    Tensor tensor;
    float scale = 0.1f;
    int zero_point = 0;
    
    QTensor quant = Quantization::quantize(tensor, scale, zero_point);
    Tensor dequant = Quantization::dequantize(quant, scale, zero_point);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
