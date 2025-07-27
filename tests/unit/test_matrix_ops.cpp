#include <gtest/gtest.h>
#include "cpl/cpl.h"

TEST(MatrixOpsTest, MatrixMultiplication) {
    const int M = 4;
    const int N = 3;
    const int K = 2;

    cpl::Matrix<float> A(M, N);
    cpl::Matrix<float> B(N, K);
    cpl::Matrix<float> C(M, K);

    // Initialize A and B with some values
    for(int i=0; i<M*N; ++i) A.data()[i] = i + 1;
    for(int i=0; i<N*K; ++i) B.data()[i] = i + 1;

    matrix_multiply(A, B, C);

    // Verify the result
    // A = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    // B = [[1, 2], [3, 4], [5, 6]]
    // C = A * B = [[22, 28], [49, 64], [76, 100], [103, 136]]

    EXPECT_FLOAT_EQ(C.at(0, 0), 22);
    EXPECT_FLOAT_EQ(C.at(0, 1), 28);
    EXPECT_FLOAT_EQ(C.at(1, 0), 49);
    EXPECT_FLOAT_EQ(C.at(1, 1), 64);
    EXPECT_FLOAT_EQ(C.at(2, 0), 76);
    EXPECT_FLOAT_EQ(C.at(2, 1), 100);
    EXPECT_FLOAT_EQ(C.at(3, 0), 103);
    EXPECT_FLOAT_EQ(C.at(3, 1), 136);
}

TEST(MatrixOpsTest, Transpose) {
    const int M = 2;
    const int N = 3;

    cpl::Matrix<float> A(M, N);
    cpl::Matrix<float> B(N, M);

    for(int i=0; i<M*N; ++i) A.data()[i] = i + 1;

    transpose(A, B);
    
    // A = [[1, 2, 3], [4, 5, 6]]
    // B = [[1, 4], [2, 5], [3, 6]]

    EXPECT_FLOAT_EQ(B.at(0, 0), 1);
    EXPECT_FLOAT_EQ(B.at(0, 1), 4);
    EXPECT_FLOAT_EQ(B.at(1, 0), 2);
    EXPECT_FLOAT_EQ(B.at(1, 1), 5);
    EXPECT_FLOAT_EQ(B.at(2, 0), 3);
    EXPECT_FLOAT_EQ(B.at(2, 1), 6);
} 