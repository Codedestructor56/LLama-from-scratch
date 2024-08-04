#include <iostream>
#include <sentencepiece_processor.h>
#include "tensor.h"
using namespace std;
sentencepiece::SentencePieceProcessor processor;

const auto status = processor.Load("bpe_tokenizer.model");


void print_tensor(const Tensor<FLOAT32>& tensor) {
    std::cout << tensor << std::endl;
}


void test_get_slice() {
    //all tests failing, reimplement the get_slice function incrementally
    std::vector<int> shape{3, 3, 3};
    Tensor<FLOAT32> test = Tensor<FLOAT32>::rand(shape);

    //std::cout << "Original tensor:" << std::endl;
    //print_tensor(test);

    // Test 1: Basic slice with default traversal and stride
    std::vector<int> start_indices{0, 0, 0};
    std::vector<int> end_indices{2, 2, 2};
    //Tensor<FLOAT32> slice1 = test.get_slice(start_indices, end_indices);
    //std::cout << "Test 1: Basic slice" << std::endl;
    //print_tensor(slice1);

    // Test 3: Slice with stride
    std::vector<int> stride{2, 2, 2};
    //Tensor<FLOAT32> slice3 = test.get_slice(start_indices, end_indices, 1, stride);
    //std::cout << "Test 3: Slice with stride" << std::endl;
    //print_tensor(slice3);

    // Test 4: Slice with -1 end index
    std::vector<int> end_indices_full{-1, -1, -1};
    //Tensor<FLOAT32> slice4 = test.get_slice(start_indices, end_indices_full);
    //std::cout << "Test 4: Slice with -1 end index" << std::endl;
    //print_tensor(slice4);

    // Test 5: Mixed slice with different start, end, and stride
    std::vector<int> mixed_start{1, 0, 1};
    std::vector<int> mixed_end{3, 2, 3};
    std::vector<int> mixed_stride{1, 1, 2};
    //Tensor<FLOAT32> slice5 = test.get_slice(mixed_start, mixed_end, 1, mixed_stride);
    //std::cout << "Test 5: Mixed slice with different start, end, and stride" << std::endl;
    //print_tensor(slice5);

    // Test 6: Edge case with invalid indices (should throw an error)
    try {
        std::vector<int> invalid_start{0, 0, 0};
        std::vector<int> invalid_end{4, 4, 4}; // Out of bounds
        //Tensor<FLOAT32> slice6 = test.get_slice(invalid_start, invalid_end);
        //std::cout << "Test 6: Invalid indices slice (should throw an error)" << std::endl;
        //print_tensor(slice6);
    } catch (const std::runtime_error& e) {
        //std::cout << "Test 6: Caught expected runtime error: " << e.what() << std::endl;
    }
}

int main(){
  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    // error
  }


  std::vector<std::string> pieces;
  processor.Encode("This is a test.", &pieces);
  for (const std::string &token : pieces) {
    std::cout << token << std::endl;
  }

  std::vector<int> ids;
  processor.Encode("This is a test.", &ids);
  for (const int id : ids) {
    std::cout << id << std::endl;
  }
  Tensor<FLOAT16> tens= Tensor<FLOAT16>();

  vector<int> shape{1,3,3};
  vector<int> index{0,1,2};

  vector<int> data{1,1,1,2,2,2,3,3,3};
  Tensor<FLOAT32> test = Tensor<FLOAT32>::rand(shape);
  //cout<<test<<endl;
  //cout<<test.get(index)<<endl;
  //test.set(index, static_cast<float>(3));
  //cout<<test<<endl;
  Tensor<UINT8> test1 = Tensor<UINT8>(data, shape);
  cout<<test1<<endl; 
  test_get_slice();
}
