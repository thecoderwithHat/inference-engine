#include <gtest/gtest.h>
#include "core/tensor.h"

using namespace infer;

TEST(TensorTest, CreationAndAccess) {
    Tensor tensor;
    EXPECT_NE(tensor.data(), nullptr);
}

TEST(TensorTest, ShapeHandling) {
    Tensor tensor;
    auto shape = tensor.shape();
    EXPECT_GE(shape.size(), 0);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
