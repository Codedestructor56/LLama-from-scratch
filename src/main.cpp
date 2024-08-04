#include <iostream>
#include <sentencepiece_processor.h>
#include "tensor.h"
using namespace std;
sentencepiece::SentencePieceProcessor processor;

const auto status = processor.Load("bpe_tokenizer.model");


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
}
