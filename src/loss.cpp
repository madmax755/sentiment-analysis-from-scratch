#include "tensor3d.hpp"
#include "loss.hpp"

float CrossEntropyLoss::compute(const Tensor3D& predicted, const Tensor3D& target) const {
    float loss = 0.0f;
    for (size_t i = 0; i < predicted.height; ++i) {
        for (size_t j = 0; j < predicted.width; ++j) {
            // add small epsilon (1e-7) to prevent log(0)
            loss -= target(0, i, j) * std::log(predicted(0, i, j) + 1e-7f);
        }
    }
    return loss / predicted.width;  // average loss over batch
}

Tensor3D CrossEntropyLoss::derivative(const Tensor3D& predicted, const Tensor3D& target) const {
    // when combined with softmax output, gradient simplifies to (predicted - target)
    // this is because d(cross_entropy)/d(softmax_input) = predicted - target
    return predicted - target;
}


float MSELoss::compute(const Tensor3D& predicted, const Tensor3D& target) const {
    float loss = 0.0f;
    for (size_t i = 0; i < predicted.height; ++i) {
        for (size_t j = 0; j < predicted.width; ++j) {
            float dif = predicted(0, i, j) - target(0, i, j);
            loss += dif * dif;
        }
    }
    return loss / (2.0f * predicted.width);  // Average over batch and divide by 2
}

Tensor3D MSELoss::derivative(const Tensor3D& predicted, const Tensor3D& target) const {
    return (predicted - target) * (1.0f / predicted.width);
}