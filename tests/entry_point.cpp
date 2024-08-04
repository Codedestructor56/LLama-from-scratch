#include "tensor.h"
#include "get_slice.h"
#include "set_slice.h"


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Provide a test number (1 for get_slice, 2 for set_slice)" << std::endl;
        return 1;
    }

    int test_num = std::stoi(argv[1]);
    switch (test_num) {
        case 1:
            std::cout << "Running get_slice test..." << std::endl;
            test_get_slice();
            break;
        case 2:
            std::cout << "Running set_slice test..." << std::endl;
            test_set_slice();
            break;
        default:
            std::cout << "Invalid test number. Provide 1 for get_slice, 2 for set_slice" << std::endl;
            return 1;
    }

    return 0;
}
