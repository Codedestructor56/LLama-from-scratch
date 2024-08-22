#include <cassert>
#include <iostream>
#include "embeddings.h"
#include "tensor.h"


void test_embeddings() {
    // Test Parameters
    size_t vocab_size = 10;
    size_t embedding_dim = 4;

    // Instantiate Embeddings object
    Embeddings<FLOAT32> embeddings(vocab_size, embedding_dim);

    // Test Forward Method: Single Token ID
    std::vector<int> input_shape = {1, 1}; // (batch_size = 1, sequence_length = 1)
    Tensor<UINT32> input(input_shape);
    input.set({0, 0}, 3); // Token ID 3

    Tensor<FLOAT32> output = embeddings.forward(input);
    assert(output.shape[0] == 1);          // Batch size
    assert(output.shape[1] == embedding_dim); // Embedding dimension

    // Test Forward Method: Multiple Token IDs
    input_shape = {2, 1}; // (batch_size = 2, sequence_length = 1)
    Tensor<UINT32> input_multiple(input_shape);
    input_multiple.set({0, 0}, 2); // Token ID 2
    input_multiple.set({1, 0}, 7); // Token ID 7

    output = embeddings.forward(input_multiple);
    assert(output.shape[0] == 2);         // Batch size
    assert(output.shape[1] == embedding_dim); // Embedding dimension

    // Test Forward Method: Out of Range Token ID
    Tensor<UINT32> input_out_of_range({1, 1});
    input_out_of_range.set({0, 0}, 12); // Token ID out of range

    try {
        embeddings.forward(input_out_of_range);
        // If we reach this line, the test has failed because an exception should have been thrown.
        assert(false);
    } catch (const std::out_of_range& e) {
        // Expected exception due to out-of-range token ID
        std::cout << "Caught expected out-of-range exception: " << e.what() << std::endl;
    }

    std::cout << "All tests passed!" << std::endl;
}

