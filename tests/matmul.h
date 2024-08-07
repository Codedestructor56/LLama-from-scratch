#include <iostream>
#include <vector>
#include <cassert>
#include "tensor.h"

void matmul_tests() {
    try {
        // Test 1: Simple 2D matrices
        Tensor<UINT8> A({1, 2, 3, 4, 5, 6}, {2, 3});
        Tensor<UINT8> B({7, 8, 9, 10, 11, 12}, {3, 2});
        Tensor<UINT8> C = matmul(A, B);
        std::vector<uint8_t> expected1 = {58, 64, 139, 154};
        assert(std::equal(C.data(), C.data() + 4, expected1.begin()));
        
        std::cout << "\nTest 1: Simple 2D matrices\n";
        std::cout << "A:\n" << A;
        std::cout << "B:\n" << B;
        std::cout << "C:\n" << C;
        // Test 2: Matrix with identity matrix
        Tensor<UINT8> D({1, 2, 3, 4}, {2, 2});
        Tensor<UINT8> E({1, 0, 0, 1}, {2, 2}); // Identity matrix
        Tensor<UINT8> F = matmul(D, E);
        std::vector<uint8_t> expected2 = {1, 2, 3, 4};
        assert(std::equal(F.data(), F.data() + 4, expected2.begin()));
        
        std::cout << "\nTest 2: Matrix with identity matrix\n";
        std::cout << "D:\n" << D;
        std::cout << "E:\n" << E;
        std::cout << "F:\n" << F;
        // Test 3: Edge case - 1xN and Nx1 matrices
        Tensor<UINT8> G({1, 2, 3}, {1, 3});
        Tensor<UINT8> H({4, 5, 6}, {3, 1});
        Tensor<UINT8> I = matmul(G, H);
        std::vector<uint8_t> expected3 = {32};
        assert(std::equal(I.data(), I.data() + 1, expected3.begin()));
        
        std::cout << "\nTest 3: Edge case - 1xN and Nx1 matrices\n";
        std::cout << "G:\n" << G;
        std::cout << "H:\n" << H;
        std::cout << "I:\n" << I;
        // Test 4: 2x2 matrix multiplied by 2x2 matrix
        Tensor<UINT8> J({1, 2, 3, 4}, {2, 2});
        Tensor<UINT8> K({5, 6, 7, 8}, {2, 2});
        Tensor<UINT8> L = matmul(J, K);
        std::vector<uint8_t> expected4 = {19, 22, 43, 50};
        assert(std::equal(L.data(), L.data() + 4, expected4.begin()));
        

        std::cout << "\nTest 4: 2x2 matrix multiplied by 2x2 matrix\n";
        std::cout << "J:\n" << J;
        std::cout << "K:\n" << K;
        std::cout << "L:\n" << L;
        // Test 5: 3D tensors with matching last dimensions
        Tensor<UINT32> M({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {2, 2, 3});
        Tensor<UINT32> N({1, 2, 3, 4, 5, 6}, {3, 2});
        Tensor<UINT32> O = matmul(M, N);
        std::vector<uint32_t> expected5 = {22, 28, 49, 64, 76, 100, 103, 136};


        std::cout << "\nTest 5: 3D tensors with matching last dimensions\n";
        std::cout << "M:\n" << M;
        std::cout << "N:\n" << N;
        std::cout << "O:\n" << O;
        assert(std::equal(O.data(), O.data() + 8, expected5.begin()));

        // Test 6: 1x1 matrix multiplied by 1x1 matrix
        Tensor<UINT8> P({7}, {1, 1});
        Tensor<UINT8> Q({8}, {1, 1});
        Tensor<UINT8> R = matmul(P, Q);
        std::vector<uint8_t> expected6 = {56};
        assert(std::equal(R.data(), R.data() + 1, expected6.begin()));

        // Test 7: 4D tensors with simple values
        Tensor<UINT8> S({1, 2, 3, 4, 5, 6}, {1, 2, 2, 3});
        Tensor<UINT8> T({7, 8, 9, 10, 11, 12}, {3, 2});
        Tensor<UINT8> U = matmul(S, T);
        std::vector<uint8_t> expected7 = {58, 64, 139, 154, 58, 64, 139, 154};
        assert(std::equal(U.data(), U.data() + 8, expected7.begin()));

        // Test 8: Batch-wise multiplication of 2x3 and 3x2 tensors
        Tensor<UINT8> V({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {2, 2, 3});
        Tensor<UINT8> W({7, 8, 9, 10, 11, 12}, {3, 2});
        Tensor<UINT8> X = matmul(V, W);
        std::vector<uint8_t> expected8 = {58, 64, 139, 154, 58, 64, 139, 154};
        assert(std::equal(X.data(), X.data() + 8, expected8.begin()));

        // Test 9: Larger tensors
        Tensor<UINT32> Y({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {2, 2, 3});
        Tensor<UINT32> Z({1, 2, 3, 4, 5, 6}, {3, 2});
        Tensor<UINT32> A1 = matmul(Y, Z);
        std::vector<uint32_t> expected9 = {22, 28, 49, 64, 76, 100, 130, 160, 190, 232, 274, 316};
        assert(std::equal(A1.data(), A1.data() + 12, expected9.begin()));

        // Test 10: Very simple 2x2 matrices with small values
        Tensor<UINT8> B1({2, 3, 4, 5}, {2, 2});
        Tensor<UINT8> C1({1, 2, 3, 4}, {2, 2});
        Tensor<UINT8> D1 = matmul(B1, C1);
        std::vector<uint8_t> expected10 = {11, 16, 23, 34};
        assert(std::equal(D1.data(), D1.data() + 4, expected10.begin()));

        std::cout << "All tests passed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
    }
}

