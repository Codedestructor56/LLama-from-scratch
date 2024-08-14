#include "dataloader.h"
#include <numeric> 
#include <stdexcept>

Dataloader::Dataloader(const std::string& data_path, size_t batch_size)
    : data_path_(data_path),
      batch_size_(batch_size),
      stop_requested_(false) {
   
    std::string model_path = "./bpe_tokenizer.model";
    if (!tokenizer_.Load(model_path).ok()) {
        throw std::runtime_error("Failed to load SentencePiece model from " + model_path);
    }
    
    std::ifstream file("./data/" + data_path_ + ".txt");
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
        worker_threads_.emplace_back([this] {
            try {
                load_data();
            } catch (const std::exception &e) {
                std::cerr << "Exception in thread: " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "Unknown exception in thread." << std::endl;
            }
        });
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


TensorVariant wrapInVariant(const auto& batch) {
    using T = std::decay_t<decltype(batch)>;

    if constexpr (std::is_same_v<T, std::shared_ptr<Tensor<FLOAT32>>> ||
                  std::is_same_v<T, std::shared_ptr<Tensor<FLOAT16>>> ||
                  std::is_same_v<T, std::shared_ptr<Tensor<INT32>>> ||
                  std::is_same_v<T, std::shared_ptr<Tensor<UINT32>>> ||
                  std::is_same_v<T, std::shared_ptr<Tensor<INT8>>> ||
                  std::is_same_v<T, std::shared_ptr<Tensor<UINT8>>>) {
        return TensorVariant(batch);
    } else {
        throw std::runtime_error("Unsupported tensor type");
    }
}
void Dataloader::load_data() {
    std::vector<int> shape = { static_cast<int>(batch_size_) }; 
    std::vector<Tensor<UINT32>> tensors;
    Tensor<UINT32> stacked_tensors;
    
    size_t current_batch_size = 0;

    for (const std::string& text : lines) {
        std::vector<int> ids;
        tokenizer_.Encode(text, &ids);
        
        if (current_batch_size + ids.size() <= batch_size_) {
            try {
                Tensor<UINT32> tensor = text_to_tensor_ids(text, {static_cast<int>(ids.size())});
                tensors.push_back(tensor);
                current_batch_size += ids.size();
            } catch (const std::runtime_error& e) {
                std::cerr << "Error processing text: " << e.what() << std::endl;
                break;
            }
        } else {
            break;
        }

        if (current_batch_size >= batch_size_) {
            stacked_tensors = vstack(tensors);
            {
                std::lock_guard<std::mutex> lock(data_mutex_);
                current_batch_ = wrapInVariant(stacked_tensors);
            }
            tensors.clear();
            current_batch_size = 0;
        }
    }

    if (!tensors.empty()) {
        stacked_tensors = vstack(tensors);
        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            current_batch_ = wrapInVariant(stacked_tensors);
        }
    }
}

Tensor<FLOAT32> Dataloader::get_next_batch_float32() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (auto ptr = std::get_if<std::shared_ptr<Tensor<FLOAT32>>>(&current_batch_)) {
        return **ptr;
    } else {
        throw std::bad_variant_access();
    }
}

Tensor<FLOAT16> Dataloader::get_next_batch_float16() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (auto ptr = std::get_if<std::shared_ptr<Tensor<FLOAT16>>>(&current_batch_)) {
        return **ptr;
    } else {
        throw std::bad_variant_access();
    }
}

Tensor<INT8> Dataloader::get_next_batch_int8() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (auto ptr = std::get_if<std::shared_ptr<Tensor<INT8>>>(&current_batch_)) {
        return **ptr;
    } else {
        throw std::bad_variant_access();
    }
}

Tensor<INT32> Dataloader::get_next_batch_int32() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (auto ptr = std::get_if<std::shared_ptr<Tensor<INT32>>>(&current_batch_)) {
        return **ptr;
    } else {
        throw std::bad_variant_access();
    }
}

Tensor<UINT8> Dataloader::get_next_batch_uint8() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (auto ptr = std::get_if<std::shared_ptr<Tensor<UINT8>>>(&current_batch_)) {
        return **ptr;
    } else {
        throw std::bad_variant_access();
    }
}

Tensor<UINT32> Dataloader::get_next_batch_uint32() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (auto ptr = std::get_if<std::shared_ptr<Tensor<UINT32>>>(&current_batch_)) {
        return **ptr;
    } else {
        throw std::bad_variant_access();
    }
}

