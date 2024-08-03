//build inst:
//g++ -Iexternal/sentencepiece/src -Lexternal/sentencepiece/src -o bpe_tokenizer bpe_trainer.cpp -lsentencepiece -lsentencepiece_train
#include <sentencepiece_processor.h>
#include <sentencepiece_trainer.h>
#include <iostream>

int main(int argc, char *argv[]) {
    // Check if the user provided a filename
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
        return 1;
    }

    // The input file containing the text corpus for training
    std::string input_file = argv[1];

    // Special tokens
    const std::string unk_token = "<unk>";
    const std::string eos_token = "<eos>";
    const std::string bos_token = "<bos>";
    const std::string pad_token = "<pad>";

    // Vocabulary size
    int vocab_size = 5000;

    // Training options
    std::string training_options = 
        "--input=" + input_file +
        " --model_prefix=bpe_tokenizer" +
        " --vocab_size=" + std::to_string(vocab_size) +
        " --unk_id=0" +
        " --bos_id=1" +
        " --eos_id=2" +
        " --pad_id=3" +
        " --character_coverage=1.0" +
        " --model_type=bpe";

    // Train the BPE tokenizer
    sentencepiece::SentencePieceTrainer::Train(training_options);

    std::cout << "BPE tokenizer trained successfully!" << std::endl;

    return 0;
}
