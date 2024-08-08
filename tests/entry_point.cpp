#include "tensor.h"
#include "get_slice.h"
#include "set_slice.h"
#include "set_children.h"
#include "change_dtype.h"
#include "operation_tests.h"
#include "matmul.h"
#include "reshape.h"

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
        case 3:
            std::cout << "Running set_chilren test..." << std::endl;
            test_set_children();
            break;
        case 4:
            std::cout << "Running change_dtype test..." << std::endl;
            test_change_dtype();
            break;
        case 5:
            std::cout << "Running operations test..." << std::endl;
            generate_test_cases();
            break;
        case 6:
            std::cout << "Running matmul test..." << std::endl;
            matmul_tests();
            break;
        case 7:
            std::cout << "Running reshape test..." << std::endl;
            test_reshape();
            break;

        default:
            std::cout << "Invalid test number." << std::endl;
            return 1;
    }

    return 0;
}
