#ifndef LOSS_HPP
#define LOSS_HPP
    
#include "tensor3d.hpp"

class Loss {
   public:
    virtual ~Loss() = default;

    // Compute the loss value
    virtual float compute(const Tensor3D& predicted, const Tensor3D& target) const = 0;

    // Compute the derivative of the loss with respect to the predicted values
    virtual Tensor3D derivative(const Tensor3D& predicted, const Tensor3D& target) const = 0;
};

class CrossEntropyLoss : public Loss {
   public:
    float compute(const Tensor3D& predicted, const Tensor3D& target) const override;

    Tensor3D derivative(const Tensor3D& predicted, const Tensor3D& target) const override;
};

class MSELoss : public Loss {
   public:
    float compute(const Tensor3D& predicted, const Tensor3D& target) const override;

    Tensor3D derivative(const Tensor3D& predicted, const Tensor3D& target) const override;
};

#endif  // LOSS_HPP