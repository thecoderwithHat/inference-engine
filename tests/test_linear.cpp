#include <gtest/gtest.h>
#include "kernels/linear_scalar.h"

using namespace infer;

TEST(LinearTest, ScalarLinearLayer) {
    int input_size = 4;
    int output_size = 2;
    
    float input[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float weights[] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    float bias[] = {0.0f, 0.0f};
    float output[2];
    
    linear_scalar(input, weights, bias, output, input_size, output_size);
    
    EXPECT_FLOAT_EQ(output[0], 5.0f);
    EXPECT_FLOAT_EQ(output[1], 5.0f);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
