#include "dataloader.h"
#include <sentencepiece_processor.h>

Dataloader::Dataloader(const std::string& data_path, size_t batch_size)
    : data_path_(data_path),
      batch_size_(batch_size),
      stop_requested_(false) {
   
    std::string model_path = data_path + "/tokenizer.model";
    if (!tokenizer_.Load(model_path).ok()) {
        throw std::runtime_error("Failed to load SentencePiece model from " + model_path);
    }
    
    std::ifstream file("/data/" + data_path_ + ".txt");
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << "/data/" + data_path_ + ".txt" << std::endl;
        return;
    }

    std::string line;

    while (std::getline(file, line)) {
        lines.push_back(line);
    }

    file.close();


    start_loading();
}

Dataloader::~Dataloader() {
    stop_loading(); 
}


void Dataloader::start_loading() {
    stop_requested_ = false;  
    for (size_t i = 0; i < std::thread::hardware_concurrency(); ++i) {
        worker_threads_.emplace_back(&Dataloader::load_data, this);
    }
}


void Dataloader::stop_loading() {
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        stop_requested_ = true; 
    }
   
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    worker_threads_.clear(); 
}


Tensor<UINT32> Dataloader::text_to_tensor_ids(const std::string& text, const std::vector<int>& shape) { 
    std::vector<int> ids;
    tokenizer_.Encode(text, &ids);     
    int num_elements = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());

     if (ids.size() != static_cast<size_t>(num_elements)) {
        throw std::runtime_error("The number of token IDs does not match the specified tensor shape.");
    }

    Tensor<UINT32> result = Tensor<UINT32>(ids, shape);
    return result;
}

void Dataloader::load_data() {
    std::vector<int> shape = { static_cast<int>(batch_size_) }; 
    std::vector<Tensor<UINT32>> tensors;
    Tensor<UINT32> stacked_tensors;
    for (const std::string& text : lines) {
        try {
            Tensor<UINT32> tensor = text_to_tensor_ids(text, shape);
            tensors.push_back(tensor);
        } catch (const std::runtime_error& e) {
            std::cerr << "Error processing text: " << e.what() << std::endl;
            break;
        }
    }
    stacked_tensors = vstack(tensors);
}
