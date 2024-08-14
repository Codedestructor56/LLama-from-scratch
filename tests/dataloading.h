#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "dataloader.h"

void test_dataloader() {
  try {
        // Create a Dataloader instance
        Dataloader dataloader("wikitext-2-train", 32);

        while (true) {
            // Fetch the next batch as UINT32
            Tensor<UINT32> uint32_batch = dataloader.get_next_batch_uint32();

            // Check if the batch is empty (all batches are processed)
            if (uint32_batch.size() == 0) {
                break;
            }

            // Print the shape of the current batch
            std::cout << "Batch Shape: " << std::endl;
            for(auto elem: uint32_batch.shape){
                std::cout << elem <<","<< std::endl;
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
    }
}


