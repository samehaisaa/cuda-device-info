#include <gtest/gtest.h>
#include "cpl/cpl.h"

TEST(VectorOpsTest, VectorAddition) {
    const int N = 10;
    cpl::Vector<float> A(N);
    cpl::Vector<float> B(N);
    cpl::Vector<float> C(N);

    for (int i = 0; i < N; ++i) {
        A.at(i) = i;
        B.at(i) = i * 2;
    }

    vector_add(A, B, C);

    for (int i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(C.at(i), A.at(i) + B.at(i));
    }
}

TEST(VectorOpsTest, VectorSubtraction) {
    const int N = 10;
    cpl::Vector<float> A(N);
    cpl::Vector<float> B(N);
    cpl::Vector<float> C(N);

    for (int i = 0; i < N; ++i) {
        A.at(i) = i * 2;
        B.at(i) = i;
    }

    vector_subtract(A, B, C);

    for (int i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(C.at(i), A.at(i) - B.at(i));
    }
}

TEST(VectorOpsTest, DotProduct) {
    const int N = 10;
    cpl::Vector<float> A(N);
    cpl::Vector<float> B(N);

    float expected_dot_product = 0.0f;
    for (int i = 0; i < N; ++i) {
        A.at(i) = i + 1;
        B.at(i) = i + 1;
        expected_dot_product += (i + 1) * (i + 1);
    }

    float result = dot_product(A, B);
    EXPECT_FLOAT_EQ(result, expected_dot_product);
} 