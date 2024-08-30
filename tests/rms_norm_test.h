#include "rms_norm.h"
#include "dataloader.h"
#include "embeddings.h"

void test_rmsnorm_forward() {
    // Define the shape of the input data
    std::vector<int> shape = {2, 5}; // Example shape

    // Create a dataloader instance with a dummy data path and batch size
    Dataloader dataloader("quant_mech", 2);

    // Start loading data in the background
    dataloader.start_loading();

    // Create an embeddings instance with vocab size and embedding dimension
    Embeddings<FLOAT32> embeddings(500, 5); // Example dimensions

    // Retrieve the next batch and forward through embeddings
    Tensor<UINT32> batch_data = dataloader.get_next_batch_uint32();
    Tensor<FLOAT32> embedded_data = embeddings.forward(batch_data);

    // Instantiate the RMSNorm with epsilon
    RMSNorm<FLOAT32> rmsnorm(1e-5);

    // Perform the forward operation using the embedded data
    Tensor<FLOAT32> output = rmsnorm.forward(embedded_data);

    // Print the input and output tensors for visual inspection
    std::cout << "Input Tensor:\n";
    for (int i = 0; i < embedded_data.size(); ++i) {
        std::cout << embedded_data.data()[i] << " ";
    }
    std::cout << "\nOutput Tensor:\n";
    for (int i = 0; i < output.size(); ++i) {
        std::cout << output.data()[i] << " ";
    }
    std::cout << std::endl;

    // Optional: Add assertions to verify correctness
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
    dataloader.stop_loading();
}
