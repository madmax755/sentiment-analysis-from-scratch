#include "../include/tensor3d.hpp"

int main() {

    Tensor3D tensor(3, 3, 3);
    tensor.he_initialise();
    std::cout << tensor << std::endl;

    return 0;
}