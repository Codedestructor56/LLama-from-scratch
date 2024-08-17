#include <iostream>
#include <string>
#include "dataloader.h"

void test_dataloader() {
    try {
        // Create a Dataloader instance
        Dataloader dataloader("wikitext-2-train", 64);
        int num_batch = 0;
        // Start the asynchronous data loading
        dataloader.start_loading();

        while (true) {
            // Fetch the next batch as UINT32
            Tensor<UINT32> uint32_batch = dataloader.get_next_batch_uint32();

            // Check if the batch is empty (all batches are processed)
            if (uint32_batch.shape.size() == 0) {
                break;
            }

            // Print the shape of the current batch
            std::cout << "Batch Shape: ";
            for (auto elem : uint32_batch.shape) {
                std::cout << elem << " ";
            }
            std::cout << std::endl;
            num_batch++;
 
        }

        // Stop the asynchronous data loading
        dataloader.stop_loading();
        std::cout<<num_batch<<std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
    }
}



