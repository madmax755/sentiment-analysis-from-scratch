#ifndef LOSS_HPP
#define LOSS_HPP
    
#include "Tensor3d.hpp"

class Loss {
   public:
    virtual ~Loss() = default;

    // Compute the loss value
    virtual float compute(const Tensor3d& predicted, const Tensor3d& target) const = 0;

    // Compute the derivative of the loss with respect to the predicted values
    virtual Tensor3d derivative(const Tensor3d& predicted, const Tensor3d& target) const = 0;
};

class CrossEntropyLoss : public Loss {
   public:
    float compute(const Tensor3d& predicted, const Tensor3d& target) const override;

    Tensor3d derivative(const Tensor3d& predicted, const Tensor3d& target) const override;
};

class MSELoss : public Loss {
   public:
    float compute(const Tensor3d& predicted, const Tensor3d& target) const override;

    Tensor3d derivative(const Tensor3d& predicted, const Tensor3d& target) const override;
};

#endif  // LOSS_HPP