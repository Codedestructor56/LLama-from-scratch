#include "rms_norm.h"
#include "dataloader.h"
#include "embeddings.h"

void test_rmsnorm_forward() {
    // Define the shape of the input data and create a dataloader and embeddings
    std::vector<int> shape = {2, 5}; // Example shape

    // Create a shared_ptr to a dataloader instance with a dummy data path and batch size
    std::shared_ptr<Dataloader> dataloader = std::make_shared<Dataloader>("quant_mech", 2);

    // Start loading data in the background
    dataloader->start_loading();

    // Create a shared_ptr to an embeddings instance with vocab size and embedding dimension
    std::shared_ptr<Embeddings<FLOAT32>> embeddings = std::make_shared<Embeddings<FLOAT32>>(500, 5); // Example dimensions

    // Instantiate the RMSNorm with epsilon
    RMSNorm<FLOAT32> rmsnorm(dataloader, embeddings, 1e-5);

    // Create an input tensor
    Tensor<FLOAT32> input = Tensor<FLOAT32>::rand(shape);

    // Perform the forward operation
    Tensor<FLOAT32> output = rmsnorm.forward(input);

    // Print the input and output tensors for visual inspection
    std::cout << "Input Tensor:\n";
    for (int i = 0; i < input.size(); ++i) {
        std::cout << input.data()[i] << " ";
    }
    std::cout << "\nOutput Tensor:\n";
    for (int i = 0; i < output.size(); ++i) {
        std::cout << output.data()[i] << " ";
    }
    std::cout << std::endl;

    // Optional: Add assertions to verify correctness
    // Example: Verify that the output tensor is normalized
    float sum_of_squares = 0.0f;
    for (int i = 0; i < output.size(); ++i) {
        sum_of_squares += output.data()[i] * output.data()[i];
    }
    float rms = std::sqrt(sum_of_squares / output.size());

    if (std::abs(rms - 1.0f) < 1e-3) {
        std::cout << "RMSNorm forward pass test passed!" << std::endl;
    } else {
        std::cout << "RMSNorm forward pass test failed. RMS: " << rms << std::endl;
    }

    // Stop the dataloader after the test
    dataloader->stop_loading();
}
