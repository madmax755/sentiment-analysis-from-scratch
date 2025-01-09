#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../include/tensor3d.hpp"
#include "../include/tokeniser.hpp"

// sigmoid activation function
float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// sigmoid derivative
float sigmoid_derivative(float x) {
    float s = sigmoid(x);
    return s * (1.0f - s);
}

// relu activation function
float relu(float x) { return std::max(x, 0.0f); }

// relu derivative
float relu_derivative(float x) { return (x > 0.0f) ? 1.0f : 0.0f; }

struct TrainingExample {
    std::vector<Tensor3D> sequence;
    Tensor3D target;

    friend std::ostream& operator<<(std::ostream& os, const TrainingExample& example) {
        os << "sequence length: " << example.sequence.size() << "\n";
        os << "target: \n" << example.target << "\n";
        return os;
    }
};

// gradient storage struct for GRUCell -- defined globally to avoid circular dependency
struct GRUGradients {
    Tensor3D dW_z, dU_z, db_z;  // update gate gradients
    Tensor3D dW_r, dU_r, db_r;  // reset gate gradients
    Tensor3D dW_h, dU_h, db_h;  // hidden state gradients
    size_t input_size;
    size_t hidden_size;

    GRUGradients(size_t input_size, size_t hidden_size)
        : dW_z(hidden_size, input_size),
          dU_z(hidden_size, hidden_size),
          db_z(hidden_size, 1),
          dW_r(hidden_size, input_size),
          dU_r(hidden_size, hidden_size),
          db_r(hidden_size, 1),
          dW_h(hidden_size, input_size),
          dU_h(hidden_size, hidden_size),
          db_h(hidden_size, 1),
          input_size(input_size),
          hidden_size(hidden_size) {}

    // operator overloading for addition
    GRUGradients operator+(const GRUGradients& other) const {
        if (input_size != other.input_size or hidden_size != other.hidden_size) {
            throw std::invalid_argument("GRUGradients dimensions don't match for addition");
        }
        GRUGradients result(input_size, hidden_size);
        result.dW_z = dW_z + other.dW_z;
        result.dU_z = dU_z + other.dU_z;
        result.db_z = db_z + other.db_z;
        result.dW_r = dW_r + other.dW_r;
        result.dU_r = dU_r + other.dU_r;
        result.db_r = db_r + other.db_r;
        result.dW_h = dW_h + other.dW_h;
        result.dU_h = dU_h + other.dU_h;
        result.db_h = db_h + other.db_h;
        return result;
    }

    // operator overloading for scalar multiplication
    GRUGradients operator*(float scalar) const {
        GRUGradients result(input_size, hidden_size);
        result.dW_z = dW_z * scalar;
        result.dU_z = dU_z * scalar;
        result.db_z = db_z * scalar;
        result.dW_r = dW_r * scalar;
        result.dU_r = dU_r * scalar;
        result.db_r = db_r * scalar;
        result.dW_h = dW_h * scalar;
        result.dU_h = dU_h * scalar;
        result.db_h = db_h * scalar;
        return result;
    }
};

class GRUCell {
   private:
    // store sequence of states for BPTT
    struct TimeStep {
        Tensor3D z, r, h_candidate, h;
        Tensor3D h_prev;
        Tensor3D x;

        TimeStep(size_t hidden_size, size_t input_size)
            : z(hidden_size, 1),
              r(hidden_size, 1),
              h_candidate(hidden_size, 1),
              h(hidden_size, 1),
              h_prev(hidden_size, 1),
              x(input_size, 1) {}
    };
    std::vector<TimeStep> time_steps;

    // clear the stored states
    void clear_states() { time_steps.clear(); }

    // store the gstdradients and the gradient of the hidden state from the previous timestep
    struct BackwardResult {
        GRUGradients grads;
        Tensor3D dh_prev;

        BackwardResult(GRUGradients g, Tensor3D h) : grads(g), dh_prev(h) {}
    };

    // compute gradients for a single timestep
    BackwardResult backward(const Tensor3D& delta_h_t, size_t t) {
        if (t >= time_steps.size()) {
            throw std::runtime_error("Time step index out of bounds");
        }

        // get the stored states for this timestep
        const TimeStep& step = time_steps[t];

        // initialise gradients for this timestep
        GRUGradients timestep_grads(input_size, hidden_size);

        // 1. Hidden state gradients
        Tensor3D one_matrix(delta_h_t.height, delta_h_t.width);
        for (size_t i = 0; i < one_matrix.height; i++) {
            for (size_t j = 0; j < one_matrix.width; j++) {
                one_matrix(0, i, j) = 1.0f;
            }
        }

        Tensor3D dh_tilde = delta_h_t.hadamard(one_matrix - step.z);
        Tensor3D dz = delta_h_t.hadamard(step.h_prev - step.h_candidate);

        // 2. Candidate state gradients
        Tensor3D dg = dh_tilde.hadamard(step.h_candidate.apply([](float x) { return 1.0f - x * x; }));  // tanh derivative

        timestep_grads.dW_h = dg * step.x.transpose();
        timestep_grads.dU_h = dg * (step.r.hadamard(step.h_prev)).transpose();
        timestep_grads.db_h = dg;

        Tensor3D dx_t = timestep_grads.dW_h.transpose() * dg;
        Tensor3D dr = (timestep_grads.dU_h.transpose() * dg).hadamard(step.h_prev);
        Tensor3D dh_prev = (timestep_grads.dU_h.transpose() * dg).hadamard(step.r);

        // 3. Reset gate gradients
        Tensor3D dr_total = dr.hadamard(step.r.apply(sigmoid_derivative));

        timestep_grads.dW_r = dr_total * step.x.transpose();
        timestep_grads.dU_r = dr_total * step.h_prev.transpose();
        timestep_grads.db_r = dr_total;

        dx_t = dx_t + timestep_grads.dW_r.transpose() * dr_total;
        dh_prev = dh_prev + timestep_grads.dU_r.transpose() * dr_total;

        // 4. Update gate gradients
        Tensor3D dz_total = dz.hadamard(step.z.apply(sigmoid_derivative));

        timestep_grads.dW_z = dz_total * step.x.transpose();
        timestep_grads.dU_z = dz_total * step.h_prev.transpose();
        timestep_grads.db_z = dz_total;

        dx_t = dx_t + timestep_grads.dW_z.transpose() * dz_total;
        dh_prev = dh_prev + timestep_grads.dU_z.transpose() * dz_total;

        // 5. Final hidden state contribution
        dh_prev = dh_prev + delta_h_t.hadamard(step.z);

        // return both values
        return BackwardResult(timestep_grads, dh_prev);
    }

   public:
    // gate weights and biases
    Tensor3D W_z;  // update gate weights for input
    Tensor3D U_z;  // update gate weights for hidden state
    Tensor3D b_z;  // update gate bias

    Tensor3D W_r;  // reset gate weights for input
    Tensor3D U_r;  // reset gate weights for hidden state
    Tensor3D b_r;  // reset gate bias

    Tensor3D W_h;  // candidate hidden state weights for input
    Tensor3D U_h;  // candidate hidden state weights for hidden state
    Tensor3D b_h;  // candidate hidden state bias

    size_t input_size;
    size_t hidden_size;

    GRUCell(size_t input_size, size_t hidden_size)
        : input_size(input_size),
          hidden_size(hidden_size),
          W_z(hidden_size, input_size),
          U_z(hidden_size, hidden_size),
          b_z(hidden_size, 1),
          W_r(hidden_size, input_size),
          U_r(hidden_size, hidden_size),
          b_r(hidden_size, 1),
          W_h(hidden_size, input_size),
          U_h(hidden_size, hidden_size),
          b_h(hidden_size, 1) {
        // initialise weights using Xavier initialization
        W_z.xavier_initialise();
        U_z.xavier_initialise();
        W_r.xavier_initialise();
        U_r.xavier_initialise();
        W_h.xavier_initialise();
        U_h.xavier_initialise();

        // biases are initialised to zero by default
    }

    // get final hidden state
    Tensor3D get_last_hidden_state() const {
        if (time_steps.empty()) {
            throw std::runtime_error("No hidden state available - run forward pass first");
        }
        return time_steps.back().h;
    }

    // resets gradients in GRUGradients struct to zero
    void reset_gradients(GRUGradients& grads) {
        grads.dW_z.zero_initialise();
        grads.dU_z.zero_initialise();
        grads.db_z.zero_initialise();
        grads.dW_r.zero_initialise();
        grads.dU_r.zero_initialise();
        grads.db_r.zero_initialise();
        grads.dW_h.zero_initialise();
        grads.dU_h.zero_initialise();
        grads.db_h.zero_initialise();
    }

    // forward pass that stores states
    Tensor3D forward(const Tensor3D& x, const Tensor3D& h_prev) {
        TimeStep step(hidden_size, input_size);
        step.x = x;
        step.h_prev = h_prev;

        // update gate
        step.z = (W_z * x + U_z * h_prev + b_z).apply(sigmoid);

        // reset gate
        step.r = (W_r * x + U_r * h_prev + b_r).apply(sigmoid);

        // candidate hidden state
        step.h_candidate = (W_h * x + U_h * (step.r.hadamard(h_prev)) + b_h).apply(std::tanh);

        // final hidden state
        step.h = step.z.hadamard(h_prev) + (step.z.apply([](float x) { return 1.0f - x; }).hadamard(step.h_candidate));

        time_steps.push_back(step);
        return step.h;
    }

    GRUGradients backpropagate(const Tensor3D& final_gradient) {
        Tensor3D dh_next = final_gradient;
        GRUGradients accumulated_grads(input_size, hidden_size);
        reset_gradients(accumulated_grads);

        // backpropagate through time
        for (int t = time_steps.size() - 1; t >= 0; --t) {
            BackwardResult result = backward(dh_next, t);
            dh_next = result.dh_prev;

            // accumulate gradients for update gate
            accumulated_grads.dW_z = accumulated_grads.dW_z + result.grads.dW_z;
            accumulated_grads.dU_z = accumulated_grads.dU_z + result.grads.dU_z;
            accumulated_grads.db_z = accumulated_grads.db_z + result.grads.db_z;

            // accumulate gradients for reset gate
            accumulated_grads.dW_r = accumulated_grads.dW_r + result.grads.dW_r;
            accumulated_grads.dU_r = accumulated_grads.dU_r + result.grads.dU_r;
            accumulated_grads.db_r = accumulated_grads.db_r + result.grads.db_r;

            // accumulate gradients for candidate hidden state
            accumulated_grads.dW_h = accumulated_grads.dW_h + result.grads.dW_h;
            accumulated_grads.dU_h = accumulated_grads.dU_h + result.grads.dU_h;
            accumulated_grads.db_h = accumulated_grads.db_h + result.grads.db_h;
        }

        clear_states();
        return accumulated_grads;
    }
};

// layer class representing a single layer in the neural network
class Layer {
   public:
    Tensor3D weights;
    Tensor3D bias;
    std::string activation_function;

    /**
     * @brief Constructs a Layer object with specified input size, output size, and activation function.
     * @param input_size The number of input neurons.
     * @param output_size The number of output neurons.
     * @param activation_function The activation function to use (default: "sigmoid").
     */
    Layer(size_t input_size, size_t output_size, std::string activation_function = "sigmoid")
        : weights(output_size, input_size), bias(output_size, 1), activation_function(activation_function) {
        if (activation_function == "sigmoid") {
            weights.xavier_initialise();
        } else if (activation_function == "relu") {
            weights.he_initialise();
        } else {
            weights.uniform_initialise();
        }
    }

    /**
     * @brief Performs feedforward operation for this layer.
     * @param input The input matrix.
     * @return The output matrix after applying the layer's transformation.
     */
    Tensor3D feedforward(const Tensor3D& input) {
        Tensor3D z = weights * input + bias;
        Tensor3D output(z.height, z.width);
        if (activation_function == "sigmoid") {
            output = z.apply(sigmoid);
        } else if (activation_function == "relu") {
            output = z.apply(relu);
        } else if (activation_function == "softmax") {
            output = z.softmax();
        } else {
            throw std::runtime_error("no activation function found for layer");
        }

        return output;
    }

    /**
     * @brief Performs feedforward operation for this layer and returns both output and pre-activation.
     * @param input The input matrix.
     * @return A vector containing the output matrix and pre-activation matrix.
     */
    std::vector<Tensor3D> feedforward_backprop(const Tensor3D& input) const {
        Tensor3D z = weights * input + bias;
        Tensor3D output(z.height, z.width);
        if (activation_function == "sigmoid") {
            output = z.apply(sigmoid);
        } else if (activation_function == "relu") {
            output = z.apply(relu);
        } else if (activation_function == "softmax") {
            output = z.softmax();
        } else {
            throw std::runtime_error("no activation function found for layer");
        }

        return {output, z};
    }
};

// -----------------------------------------------------------------------------------------------------
// ---------------------------------------- LOSS FUNCTIONS ----------------------------------------------

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
    float compute(const Tensor3D &predicted, const Tensor3D &target) const override {
        float loss = 0.0f;
        for (size_t i = 0; i < predicted.height; ++i) {
            for (size_t j = 0; j < predicted.width; ++j) {
                // add small epsilon (1e-10) to prevent log(0)
                loss -= target(0, i, j) * std::log(predicted(0, i, j) + 1e-10f);
            }
        }
        return loss / predicted.width;  // average loss over batch
    }

    Tensor3D derivative(const Tensor3D &predicted, const Tensor3D &target) const override {
        // when combined with softmax output, gradient simplifies to (predicted - target)
        // this is because d(cross_entropy)/d(softmax_input) = predicted - target
        return predicted - target;
    }
};

class MSELoss : public Loss {
   public:
    float compute(const Tensor3D& predicted, const Tensor3D& target) const override {
        float loss = 0.0f;
        for (size_t i = 0; i < predicted.height; ++i) {
            for (size_t j = 0; j < predicted.width; ++j) {
                float dif = predicted(0, i, j) - target(0, i, j);
                loss += dif * dif;
            }
        }
        return loss / (2.0f * predicted.width);  // Average over batch and divide by 2
    }

    Tensor3D derivative(const Tensor3D& predicted, const Tensor3D& target) const override {
        return (predicted - target) * (1.0f / predicted.width);
    }
};

// -----------------------------------------------------------------------------------------------------
// ---------------------------------------- OPTIMISERS -------------------------------------------------

class MLPOptimiser {
   public:
    /**
     * @brief Computes and applies updates to the network layers based on gradients.
     * @param layers The layers of the neural network to update.
     * @param gradients The gradients used for updating the layers.
     */
    virtual void compute_and_apply_updates(std::vector<Layer>& layers, const std::vector<std::vector<Tensor3D>>& gradients) = 0;

    /**
     * @brief Virtual destructor for the MLPOptimiser class.
     */
    virtual ~MLPOptimiser() = default;

    struct GradientResult {
        std::vector<std::vector<Tensor3D>>
            gradients;                  // list of layers, each layer has a list of weight and bias gradient matrices
        Tensor3D input_layer_gradient;  // gradient of the input layer - for more general use as parts of bigger architectures
        Tensor3D output;                // output of the network
    };

    /**
     * @brief Calculates gradients for a single example.
     * @param layers The layers of the neural network.
     * @param input The input matrix.
     * @param target The target matrix.
     * @return A GradientResult struct containing gradients and the output of the network.
     */
    virtual GradientResult calculate_gradient(const std::vector<Layer>& layers, const Tensor3D& input, const Tensor3D& target,
                                              const Loss& loss) {
        // forward pass
        std::vector<Tensor3D> activations = {input};
        std::vector<Tensor3D> preactivations = {input};

        for (const auto& layer : layers) {
            auto results = layer.feedforward_backprop(activations.back());
            activations.push_back(results[0]);
            preactivations.push_back(results[1]);
        }

        // backward pass
        int num_layers = layers.size();
        std::vector<Tensor3D> deltas;
        deltas.reserve(num_layers);

        // output layer error (δ^L = ∇_a C ⊙ σ'(z^L))
        Tensor3D output_delta = loss.derivative(activations.back(), target);
        if (layers.back().activation_function == "sigmoid") {
            output_delta = output_delta.hadamard(preactivations.back().apply(sigmoid_derivative));
        } else if (layers.back().activation_function == "relu") {
            output_delta = output_delta.hadamard(preactivations.back().apply(relu_derivative));
        } else if (layers.back().activation_function == "softmax" or layers.back().activation_function == "none") {
            // for softmax and none, the delta is already correct (assuming cross-entropy loss)
        } else {
            throw std::runtime_error("Unsupported activation function");
        }
        deltas.push_back(output_delta);

        // hidden layer errors (δ^l = ((w^(l+1))^T δ^(l+1)) ⊙ σ'(z^l))
        for (int l = num_layers - 2; l >= 0; --l) {
            Tensor3D delta = (layers[l + 1].weights.transpose() * deltas.back());
            if (layers[l].activation_function == "sigmoid") {
                delta = delta.hadamard(preactivations[l + 1].apply(sigmoid_derivative));
            } else if (layers[l].activation_function == "relu") {
                delta = delta.hadamard(preactivations[l + 1].apply(relu_derivative));
            } else if (layers[l].activation_function == "none") {
                // delta = delta
            } else {
                throw std::runtime_error("Unsupported activation function");
            }
            deltas.push_back(delta);
        }

        // reverse deltas to match layer order
        std::reverse(deltas.begin(), deltas.end());

        // calculate gradients
        std::vector<std::vector<Tensor3D>> gradients;
        for (int l = 0; l < num_layers; ++l) {
            Tensor3D weight_gradient = deltas[l] * activations[l].transpose();
            gradients.push_back({weight_gradient, deltas[l]});
        }

        // as we don't treat the input as a layer, we need to return the input layer errors separately
        Tensor3D input_delta = layers[0].weights.transpose() * deltas[0];

        // return a GradientResult struct for purposes of tracking loss
        return {gradients, input_delta, activations.back()};
    }

    /**
     * @brief Averages gradients from multiple examples.
     * @param batch_gradients A vector of gradients from multiple examples.
     * @return The averaged gradients.
     */
    std::vector<std::vector<Tensor3D>> average_gradients(const std::vector<std::vector<std::vector<Tensor3D>>>& batch_gradients) {
        std::vector<std::vector<Tensor3D>> avg_gradients;
        size_t num_layers = batch_gradients[0].size();
        size_t batch_size = batch_gradients.size();

        for (size_t l = 0; l < num_layers; ++l) {
            Tensor3D avg_weight_grad(batch_gradients[0][l][0].height, batch_gradients[0][l][0].width);
            Tensor3D avg_bias_grad(batch_gradients[0][l][1].height, batch_gradients[0][l][1].width);

            for (const auto& example_gradients : batch_gradients) {
                avg_weight_grad = avg_weight_grad + example_gradients[l][0];
                avg_bias_grad = avg_bias_grad + example_gradients[l][1];
            }

            avg_weight_grad = avg_weight_grad * (1.0f / batch_size);
            avg_bias_grad = avg_bias_grad * (1.0f / batch_size);

            avg_gradients.push_back({avg_weight_grad, avg_bias_grad});
        }

        return avg_gradients;
    }
};

class MLPSGDOptimiser : public MLPOptimiser {
   private:
    float learning_rate;
    std::vector<std::vector<Tensor3D>> velocity;

   public:
    /**
     * @brief Constructs an MLPSGDOptimiser object with the specified learning rate.
     * @param lr The learning rate (default: 0.1f).
     */
    MLPSGDOptimiser(float lr = 0.1f) : learning_rate(lr) {}

    /**
     * @brief Initializes the velocity vectors for SGD optimization.
     * @param layers The layers of the neural network.
     */
    void initialise_velocity(const std::vector<Layer>& layers) {
        velocity.clear();
        for (const auto& layer : layers) {
            velocity.push_back(
                {Tensor3D(layer.weights.height, layer.weights.width), Tensor3D(layer.bias.height, layer.bias.width)});
        }
    }

    /**
     * @brief Computes and applies updates using Stochastic Gradient Descent.
     * @param layers The layers of the neural network to update.
     * @param gradients The gradients used for updating the layers.
     */
    void compute_and_apply_updates(std::vector<Layer>& layers, const std::vector<std::vector<Tensor3D>>& gradients) override {
        if (velocity.empty()) {
            initialise_velocity(layers);
        }

        // compute and apply updates
        for (size_t l = 0; l < layers.size(); ++l) {
            for (int i = 0; i < 2; ++i) {  // 0 for weights, 1 for biases
                // compute adjustment
                velocity[l][i] = gradients[l][i] * -learning_rate;
            }
            // apply adjustment
            layers[l].weights = layers[l].weights + velocity[l][0];
            layers[l].bias = layers[l].bias + velocity[l][1];
        }
    }
};

class MLPSGDMomentumOptimiser : public MLPOptimiser {
   private:
    float learning_rate;
    float momentum;
    std::vector<std::vector<Tensor3D>> velocity;

   public:
    /**
     * @brief Constructs an MLPSGDMomentumOptimiser object with the specified learning rate and momentum.
     * @param lr The learning rate (default: 0.1f).
     * @param mom The momentum coeficient (default: 0.9f).
     */
    MLPSGDMomentumOptimiser(float lr = 0.1f, float mom = 0.9f) : learning_rate(lr), momentum(mom) {}

    /**
     * @brief Initializes the velocity vectors for SGD with Momentum optimization.
     * @param layers The layers of the neural network.
     */
    void initialise_velocity(const std::vector<Layer>& layers) {
        velocity.clear();
        for (const auto& layer : layers) {
            velocity.push_back(
                {Tensor3D(layer.weights.height, layer.weights.width), Tensor3D(layer.bias.height, layer.bias.width)});
        }
    }

    /**
     * @brief Computes and applies updates using Stochastic Gradient Descent with Momentum.
     * @param layers The layers of the neural network to update.
     * @param gradients The gradients used for updating the layers.
     */
    void compute_and_apply_updates(std::vector<Layer>& layers, const std::vector<std::vector<Tensor3D>>& gradients) override {
        if (velocity.empty()) {
            initialise_velocity(layers);
        }

        // compute updates
        for (size_t l = 0; l < layers.size(); ++l) {
            for (int i = 0; i < 2; ++i) {  // 0 for weights, 1 for biases
                // compute adjustments
                velocity[l][i] = (velocity[l][i] * momentum) - (gradients[l][i] * learning_rate);
            }
            // apply adjustments
            layers[l].weights = layers[l].weights + velocity[l][0];
            layers[l].bias = layers[l].bias + velocity[l][1];
        }
    }
};

class MLPAdamOptimiser : public MLPOptimiser {
   private:
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    int t;                                 // timestep
    std::vector<std::vector<Tensor3D>> m;  // first moment
    std::vector<std::vector<Tensor3D>> v;  // second moment

   public:
    /**
     * @brief Constructs an MLPAdamOptimiser object with the specified parameters.
     * @param lr The learning rate (default: 0.0f01).
     * @param b1 The beta1 parameter (default: 0.9f).
     * @param b2 The beta2 parameter (default: 0.9f99).
     * @param eps The epsilon parameter for numerical stability (default: 1e-8).
     */
    MLPAdamOptimiser(float lr = 0.001f, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f)
        : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), t(0) {}

    /**
     * @brief Initializes the first and second moment vectors for Adam optimization.
     * @param layers The layers of the neural network.
     */
    void initialise_moments(const std::vector<Layer>& layers) {
        m.clear();
        v.clear();
        m.reserve(layers.size());
        v.reserve(layers.size());
        for (const auto& layer : layers) {
            m.push_back({Tensor3D(layer.weights.height, layer.weights.width), Tensor3D(layer.bias.height, layer.bias.width)});
            v.push_back({Tensor3D(layer.weights.height, layer.weights.width), Tensor3D(layer.bias.height, layer.bias.width)});
        }
    }

    /**
     * @brief Computes and applies updates using the Adam optimization algorithm.
     * @param layers The layers of the neural network to update.
     * @param gradients The gradients used for updating the layers.
     */
    void compute_and_apply_updates(std::vector<Layer>& layers, const std::vector<std::vector<Tensor3D>>& gradients) override {
        if (m.empty() or v.empty()) {
            initialise_moments(layers);
        }

        t++;  // increment timestep

        for (size_t l = 0; l < layers.size(); ++l) {
            for (int i = 0; i < 2; ++i) {  // 0 for weights, 1 for biases
                // update biased first moment estimate
                m[l][i] = m[l][i] * beta1 + gradients[l][i] * (1.0f - beta1);

                // update biased second raw moment estimate
                v[l][i] = v[l][i] * beta2 + gradients[l][i].hadamard(gradients[l][i]) * (1.0f - beta2);

                // compute bias-corrected first moment estimate
                Tensor3D m_hat = m[l][i] * (1.0f / (1.0f - std::pow(beta1, t)));

                // compute bias-corrected second raw moment estimate
                Tensor3D v_hat = v[l][i] * (1.0f / (1.0f - std::pow(beta2, t)));

                // compute the update
                Tensor3D update = m_hat.hadamard(v_hat.apply([this](float x) { return 1.0f / (std::sqrt(x) + epsilon); }));

                // apply the update
                if (i == 0) {
                    layers[l].weights = layers[l].weights - update * learning_rate;
                } else {
                    layers[l].bias = layers[l].bias - update * learning_rate;
                }
            }
        }
    }
};

class MLPAdamWOptimiser : public MLPOptimiser {
   private:
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    float weight_decay;
    int t;                                 // timestep
    std::vector<std::vector<Tensor3D>> m;  // first moment
    std::vector<std::vector<Tensor3D>> v;  // second moment

   public:
    /**
     * @brief Constructs an MLPAdamWOptimiser object with the specified parameters.
     * @param lr The learning rate (default: 0.0f01).
     * @param b1 The beta1 parameter (default: 0.9f).
     * @param b2 The beta2 parameter (default: 0.9f99).
     * @param eps The epsilon parameter for numerical stability (default: 1e-8).
     * @param wd The weight decay parameter (default: 0.0f1).
     */
    MLPAdamWOptimiser(float lr = 0.001f, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f, float wd = 0.001f)
        : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), weight_decay(wd), t(0) {}

    /**
     * @brief Initializes the first and second moment vectors for AdamW optimization.
     * @param layers The layers of the neural network.
     */
    void initialise_moments(const std::vector<Layer>& layers) {
        m.clear();
        v.clear();
        m.reserve(layers.size());
        v.reserve(layers.size());
        for (const auto& layer : layers) {
            m.push_back({Tensor3D(layer.weights.height, layer.weights.width), Tensor3D(layer.bias.height, layer.bias.width)});
            v.push_back({Tensor3D(layer.weights.height, layer.weights.width), Tensor3D(layer.bias.height, layer.bias.width)});
        }
    }

    /**
     * @brief Computes and applies updates using the AdamW optimization algorithm.
     * @param layers The layers of the neural network to update.
     * @param gradients The gradients used for updating the layers.
     */
    void compute_and_apply_updates(std::vector<Layer>& layers, const std::vector<std::vector<Tensor3D>>& gradients) override {
        if (m.empty() || v.empty()) {
            initialise_moments(layers);
        }

        t++;  // increment timestep

        for (size_t l = 0; l < layers.size(); ++l) {
            for (int i = 0; i < 2; ++i) {  // 0 for weights, 1 for biases
                // update biased first moment estimate
                m[l][i] = m[l][i] * beta1 + gradients[l][i] * (1.0f - beta1);

                // update biased second raw moment estimate
                v[l][i] = v[l][i] * beta2 + gradients[l][i].hadamard(gradients[l][i]) * (1.0f - beta2);

                // compute bias-corrected first moment estimate
                Tensor3D m_hat = m[l][i] * (1.0f / (1.0f - std::pow(beta1, t)));

                // compute bias-corrected second raw moment estimate
                Tensor3D v_hat = v[l][i] * (1.0f / (1.0f - std::pow(beta2, t)));

                // compute the Adam update
                Tensor3D update = m_hat.hadamard(v_hat.apply([this](float x) { return 1.0f / (std::sqrt(x) + epsilon); }));

                // apply the update
                if (i == 0) {  // for weights
                    // apply weight decay
                    layers[l].weights = layers[l].weights * (1.0f - learning_rate * weight_decay);
                    // apply Adam update
                    layers[l].weights = layers[l].weights - (update * learning_rate);
                } else {  // for biases
                    // biases typically don't use weight decay
                    layers[l].bias = layers[l].bias - update * learning_rate;
                }
            }
        }
    }
};

class GRUOptimiser {
   public:
    virtual void compute_and_apply_updates(GRUCell& gru, const GRUGradients& gradients) = 0;
    virtual ~GRUOptimiser() = default;
};

class GRUSGDOptimiser : public GRUOptimiser {
   private:
    float learning_rate;

   public:
    GRUSGDOptimiser(float lr = 0.1f) : learning_rate(lr) {}

    void compute_and_apply_updates(GRUCell& gru, const GRUGradients& grads) override {
        // update weights and biases using gradients
        gru.W_z = gru.W_z - grads.dW_z * learning_rate;
        gru.U_z = gru.U_z - grads.dU_z * learning_rate;
        gru.b_z = gru.b_z - grads.db_z * learning_rate;

        gru.W_r = gru.W_r - grads.dW_r * learning_rate;
        gru.U_r = gru.U_r - grads.dU_r * learning_rate;
        gru.b_r = gru.b_r - grads.db_r * learning_rate;

        gru.W_h = gru.W_h - grads.dW_h * learning_rate;
        gru.U_h = gru.U_h - grads.dU_h * learning_rate;
        gru.b_h = gru.b_h - grads.db_h * learning_rate;
    }
};

class GRUSGDMomentumOptimiser : public GRUOptimiser {
   private:
    float learning_rate;
    float momentum;
    GRUGradients velocity;

   public:
    GRUSGDMomentumOptimiser(float lr = 0.1f, float mom = 0.9f)
        : learning_rate(lr), momentum(mom), velocity(0, 0) {}  // sizes will be set on first use

    void compute_and_apply_updates(GRUCell& gru, const GRUGradients& grads) override {
        // initialise velocity if needed
        if (velocity.dW_z.height == 0) {
            velocity = GRUGradients(gru.input_size, gru.hidden_size);
        }

        // update velocities and apply updates for each parameter
        velocity.dW_z = velocity.dW_z * momentum - grads.dW_z * learning_rate;
        velocity.dU_z = velocity.dU_z * momentum - grads.dU_z * learning_rate;
        velocity.db_z = velocity.db_z * momentum - grads.db_z * learning_rate;

        velocity.dW_r = velocity.dW_r * momentum - grads.dW_r * learning_rate;
        velocity.dU_r = velocity.dU_r * momentum - grads.dU_r * learning_rate;
        velocity.db_r = velocity.db_r * momentum - grads.db_r * learning_rate;

        velocity.dW_h = velocity.dW_h * momentum - grads.dW_h * learning_rate;
        velocity.dU_h = velocity.dU_h * momentum - grads.dU_h * learning_rate;
        velocity.db_h = velocity.db_h * momentum - grads.db_h * learning_rate;

        // apply updates
        gru.W_z = gru.W_z + velocity.dW_z;
        gru.U_z = gru.U_z + velocity.dU_z;
        gru.b_z = gru.b_z + velocity.db_z;

        gru.W_r = gru.W_r + velocity.dW_r;
        gru.U_r = gru.U_r + velocity.dU_r;
        gru.b_r = gru.b_r + velocity.db_r;

        gru.W_h = gru.W_h + velocity.dW_h;
        gru.U_h = gru.U_h + velocity.dU_h;
        gru.b_h = gru.b_h + velocity.db_h;
    }
};

class GRUAdamOptimiser : public GRUOptimiser {
   private:
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    int t;
    GRUGradients m;
    GRUGradients v;

    void update_parameter(Tensor3D& param, Tensor3D& m_param, Tensor3D& v_param, const Tensor3D& grad) {
        // Update biased first moment estimate
        m_param = m_param * beta1 + grad * (1.0f - beta1);

        // Update biased second raw moment estimate
        v_param = v_param * beta2 + grad.hadamard(grad) * (1.0f - beta2);

        // Compute bias-corrected first moment estimate
        Tensor3D m_hat = m_param * (1.0f / (1.0f - std::pow(beta1, t)));

        // Compute bias-corrected second raw moment estimate
        Tensor3D v_hat = v_param * (1.0f / (1.0f - std::pow(beta2, t)));

        // Update parameters
        Tensor3D denom = v_hat.apply([this](float x) { return 1.0f / (std::sqrt(x) + epsilon); });
        Tensor3D update = m_hat.hadamard(denom);
        param = param - update * learning_rate;
    }

   public:
    GRUAdamOptimiser(float lr = 0.001f, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f)
        : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), t(0), m(0, 0), v(0, 0) {}

    void compute_and_apply_updates(GRUCell& gru, const GRUGradients& grads) override {
        if (m.dW_z.height == 0) {
            m = GRUGradients(gru.input_size, gru.hidden_size);
            v = GRUGradients(gru.input_size, gru.hidden_size);
        }

        t++;

        // Update all parameters
        update_parameter(gru.W_z, m.dW_z, v.dW_z, grads.dW_z);
        update_parameter(gru.U_z, m.dU_z, v.dU_z, grads.dU_z);
        update_parameter(gru.b_z, m.db_z, v.db_z, grads.db_z);
        update_parameter(gru.W_r, m.dW_r, v.dW_r, grads.dW_r);
        update_parameter(gru.U_r, m.dU_r, v.dU_r, grads.dU_r);
        update_parameter(gru.b_r, m.db_r, v.db_r, grads.db_r);
        update_parameter(gru.W_h, m.dW_h, v.dW_h, grads.dW_h);
        update_parameter(gru.U_h, m.dU_h, v.dU_h, grads.dU_h);
        update_parameter(gru.b_h, m.db_h, v.db_h, grads.db_h);
    }
};

class GRUAdamWOptimiser : public GRUOptimiser {
   private:
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    float weight_decay;
    int t;
    GRUGradients m;
    GRUGradients v;

    void update_parameter(Tensor3D& param, Tensor3D& m_param, Tensor3D& v_param, const Tensor3D& grad,
                          bool apply_weight_decay = true) {
        // Weight decay should be applied to the parameter directly
        if (apply_weight_decay) {
            param = param * (1.0f - learning_rate * weight_decay);
        }

        // Update biased first moment estimate
        m_param = m_param * beta1 + grad * (1.0f - beta1);

        // Update biased second raw moment estimate
        v_param = v_param * beta2 + grad.hadamard(grad) * (1.0f - beta2);

        // Compute bias-corrected first moment estimate
        Tensor3D m_hat = m_param * (1.0f / (1.0f - std::pow(beta1, t)));

        // Compute bias-corrected second raw moment estimate
        Tensor3D v_hat = v_param * (1.0f / (1.0f - std::pow(beta2, t)));

        // Update parameters
        Tensor3D denom = v_hat.apply([this](float x) { return 1.0f / (std::sqrt(x) + epsilon); });
        Tensor3D update = m_hat.hadamard(denom);
        param = param - update * learning_rate;
    }

   public:
    GRUAdamWOptimiser(float lr = 0.001f, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f, float wd = 0.001f)
        : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), weight_decay(wd), t(0), m(0, 0), v(0, 0) {}

    void compute_and_apply_updates(GRUCell& gru, const GRUGradients& grads) override {
        if (m.dW_z.height == 0) {
            m = GRUGradients(gru.input_size, gru.hidden_size);
            v = GRUGradients(gru.input_size, gru.hidden_size);
        }

        t++;

        // Update weights (with weight decay)
        update_parameter(gru.W_z, m.dW_z, v.dW_z, grads.dW_z, true);
        update_parameter(gru.U_z, m.dU_z, v.dU_z, grads.dU_z, true);
        update_parameter(gru.W_r, m.dW_r, v.dW_r, grads.dW_r, true);
        update_parameter(gru.U_r, m.dU_r, v.dU_r, grads.dU_r, true);
        update_parameter(gru.W_h, m.dW_h, v.dW_h, grads.dW_h, true);
        update_parameter(gru.U_h, m.dU_h, v.dU_h, grads.dU_h, true);

        // Update biases (without weight decay)
        update_parameter(gru.b_z, m.db_z, v.db_z, grads.db_z, false);
        update_parameter(gru.b_r, m.db_r, v.db_r, grads.db_r, false);
        update_parameter(gru.b_h, m.db_h, v.db_h, grads.db_h, false);
    }
};

// -----------------------------------------------------------------------------------------------------

class MLP {
   public:
    std::vector<Layer> layers;

    /**
     * @brief Constructs a MLP object with the specified topology and activation functions.
     * @param topology A vector specifying the number of neurons in each layer.
     * @param activation_functions A vector specifying the activation function for each layer (optional).
     */
    MLP(const std::vector<int>& topology, const std::vector<std::string> activation_functions = {}) {
        if (topology.empty()) {
            throw std::invalid_argument("Topology cannot be empty");
        }
        for (int size : topology) {
            if (size <= 0) {
                throw std::invalid_argument("Layer size must be positive");
            }
        }
        if ((activation_functions.size() + 1 != topology.size()) and (activation_functions.size() != 0)) {
            throw std::invalid_argument(
                "the size of activations_functions vector must be the same size as no. layers (ex. input)");
        } else if (activation_functions.size() == 0) {
            for (size_t i = 1; i < topology.size(); i++) {
                // do not pass in specific activation function - use the default specified in the layer constructor
                layers.emplace_back(topology[i - 1], topology[i]);
            }
        } else {
            for (size_t i = 1; i < topology.size(); i++) {
                layers.emplace_back(topology[i - 1], topology[i], activation_functions[i - 1]);
            }
        }
    }

    /**
     * @brief Performs feedforward operation through all layers of the network.
     * @param input The input matrix.
     * @return The output matrix after passing through all layers.
     */
    Tensor3D feedforward(const Tensor3D& input) {
        Tensor3D current = input;
        for (auto& layer : layers) {
            current = layer.feedforward(current);
        }
        return current;
    }

    size_t get_index_of_max_element_in_nx1_matrix(const Tensor3D& matrix) const {
        size_t index = 0;
        float max_value = matrix(1, 0, 0);
        for (size_t i = 1; i < matrix.height; ++i) {
            if (matrix(1, i, 0) > max_value) {
                index = i;
                max_value = matrix(1, i, 0);
            }
        }
        return index;
    }
};

class Predictor {
   private:
    GRUCell gru;
    MLP mlp;

    size_t input_size;
    size_t hidden_size;
    size_t output_size;

    std::unique_ptr<MLPOptimiser> mlp_optimiser;
    std::unique_ptr<GRUOptimiser> gru_optimiser;
    std::unique_ptr<Loss> loss;

   public:
    Predictor(size_t input_size, size_t hidden_size, size_t output_size, std::vector<int> mlp_topology,
              std::vector<std::string> mlp_activation_functions = {})
        : gru(input_size, hidden_size),
          mlp(mlp_topology, mlp_activation_functions),
          input_size(input_size),
          hidden_size(hidden_size),
          output_size(output_size) {}

    // set optimiser - call before training
    void set_gru_optimiser(std::unique_ptr<GRUOptimiser> new_optimiser) { gru_optimiser = std::move(new_optimiser); }
    // set optimiser - call before training
    void set_mlp_optimiser(std::unique_ptr<MLPOptimiser> new_optimiser) { mlp_optimiser = std::move(new_optimiser); }
    // set loss function - call before training
    void set_loss(std::unique_ptr<Loss> new_loss) { loss = std::move(new_loss); }

    // update GRU parameters using optimiser (can't be defined in GRUCell to avoid circular dependency)
    void update_parameters(const GRUGradients& gru_grads, const std::vector<std::vector<Tensor3D>>& mlp_grads) {
        if (!gru_optimiser) {
            throw std::runtime_error("no optimiser set");
        }
        gru_optimiser->compute_and_apply_updates(gru, gru_grads);
        mlp_optimiser->compute_and_apply_updates(mlp.layers, mlp_grads);
    }

    // process sequence and return prediction (full feedforward pass)
    Tensor3D predict(const std::vector<Tensor3D>& input_sequence) {
        Tensor3D h_t(hidden_size, 1);

        // process sequence through GRU
        for (const auto& x : input_sequence) {
            h_t = gru.forward(x, h_t);
        }

        // final linear layer
        return mlp.feedforward(h_t);
    }

    // process sequence and return final hidden state (used for backpropagation)
    Tensor3D feedforward_gru(const std::vector<Tensor3D>& input_sequence) {
        Tensor3D h_t(hidden_size, 1);

        // process sequence through GRU
        for (const auto& x : input_sequence) {
            h_t = gru.forward(x, h_t);
        }

        // final hidden state
        return h_t;
    }

    // gets the gradients for a single training example
    std::pair<GRUGradients, std::vector<std::vector<Tensor3D>>> compute_gradients(const std::vector<Tensor3D>& input_sequence,
                                                                                  const Tensor3D& target) {
        // forward pass
        Tensor3D final_hidden_state = feedforward_gru(input_sequence);

        auto [mlp_gradients, input_layer_gradient, output] =
            mlp_optimiser->calculate_gradient(mlp.layers, final_hidden_state, target, *loss);

        // backpropagate through GRU
        auto gru_gradients = gru.backpropagate(input_layer_gradient);
        return {gru_gradients, mlp_gradients};
    }

    void train(const std::vector<TrainingExample>& training_data, const std::vector<TrainingExample>& test_data, int epochs,
               int batch_size = 1) {
        // check if optimiser and loss function have been set
        if (!gru_optimiser) {
            throw std::runtime_error("no optimiser set - call set_optimiser() before training");
        }
        if (!loss) {
            throw std::runtime_error("no loss function set - call set_loss() before training");
        }

        // create a vector of indices
        std::vector<size_t> indices(training_data.size());
        std::iota(indices.begin(), indices.end(), 0);

        for (int epoch = 0; epoch < epochs; epoch++) {
            int no_examples = training_data.size();
            std::shuffle(indices.begin(), indices.end(), std::mt19937(std::random_device()()));

            for (int i = 0; i < no_examples; i += batch_size) {
                // create batch
                size_t batch_start = i;
                size_t batch_end = std::min(i + batch_size, no_examples);
                std::vector<size_t> batch_indices(indices.begin() + batch_start, indices.begin() + batch_end);

                std::vector<GRUGradients> accumulated_gru_gradients;
                std::vector<std::vector<std::vector<Tensor3D>>> accumulated_mlp_gradients;
                accumulated_gru_gradients.reserve(batch_end - batch_start);
                accumulated_mlp_gradients.reserve(batch_end - batch_start);

                for (const auto index : batch_indices) {
                    auto example = training_data[index];
                    auto [gru_gradients, mlp_gradients] = compute_gradients(example.sequence, example.target);
                    accumulated_gru_gradients.push_back(gru_gradients);
                    accumulated_mlp_gradients.push_back(mlp_gradients);
                }

                // average accumulated grugradients
                GRUGradients averaged_gru_gradients(input_size, hidden_size);
                for (int i = 0; i < batch_end - batch_start; i++) {
                    averaged_gru_gradients = averaged_gru_gradients + accumulated_gru_gradients[i];
                }
                averaged_gru_gradients = averaged_gru_gradients * (1.0f / (batch_end - batch_start));

                // average accumulated mlp gradients
                std::vector<std::vector<Tensor3D>> averaged_mlp_gradients =
                    mlp_optimiser->average_gradients(accumulated_mlp_gradients);

                // update GRU parameters using optimiser once batch gradients are averaged
                update_parameters(averaged_gru_gradients, averaged_mlp_gradients);

                std::cout << "\rBatch " << i / batch_size << "/" << no_examples / batch_size << " complete" << std::flush;
            }
            std::cout << "\rEpoch " << epoch << "/" << epochs << " complete    " << std::endl;
            auto test_metrics = evaluate(test_data);
            std::cout << test_metrics << "\n\n"<< std::endl;
        }
    }

    struct EvaluationMetrics {
        float loss;
        float accuracy;
        float precision;
        float recall;
        float f1_score;

        friend std::ostream& operator<<(std::ostream& os, const EvaluationMetrics& metrics) {
            os << "Loss: " << metrics.loss << "\n"
               << "Accuracy: " << metrics.accuracy * 100.0f << "%\n"
               << "Precision: " << metrics.precision * 100.0f << "%\n"
               << "Recall: " << metrics.recall * 100.0f << "%\n"
               << "F1 Score: " << metrics.f1_score * 100.0f << "%";
            return os;
        }
    };

    EvaluationMetrics evaluate(const std::vector<TrainingExample>& test_data) {
        EvaluationMetrics metrics = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float true_positives = 0.0f;
        float false_positives = 0.0f;
        float false_negatives = 0.0f;
        size_t total = test_data.size();

        for (const auto& example : test_data) {
            // get prediction and compute loss
            Tensor3D prediction = predict(example.sequence);
            metrics.loss += this->loss->compute(prediction, example.target);

            // get predicted and actual class (assuming binary classification)
            bool predicted_positive = prediction(0, 0, 0) > prediction(0, 1, 0);
            bool actual_positive = example.target(0, 0, 0) > example.target(0, 1, 0);

            // update metrics
            if (predicted_positive == actual_positive) {
                metrics.accuracy += 1.0f;
            }

            if (predicted_positive && actual_positive) {
                true_positives += 1.0f;
            } else if (predicted_positive && !actual_positive) {
                false_positives += 1.0f;
            } else if (!predicted_positive && actual_positive) {
                false_negatives += 1.0f;
            }
        }

        // average the loss
        metrics.loss /= total;
        
        // calculate accuracy
        metrics.accuracy /= total;

        // calculate precision
        metrics.precision = true_positives / (true_positives + false_positives + 1e-10f);

        // calculate recall
        metrics.recall = true_positives / (true_positives + false_negatives + 1e-10f);

        // calculate f1 score
        metrics.f1_score = 2.0f * (metrics.precision * metrics.recall) / 
                          (metrics.precision + metrics.recall + 1e-10f);

        return metrics;
    }
};

// reads csv file for columns headed 'review' and 'sentiment', tokenises, then returns training examples.
std::vector<TrainingExample> training_examples_from_csv(const std::string& filename, Tokeniser& tokeniser,
                                                        size_t no_examples = std::numeric_limits<size_t>::max()) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("could not open file: " + filename);
    }

    // read header
    std::string line;
    std::getline(file, line);

    // find indices of review and sentiment columns in header
    std::stringstream header_stream(line);
    std::string field;
    int review_idx = -1;
    int sentiment_idx = -1;
    int col_idx = 0;

    // split header on commas and find required columns
    while (std::getline(header_stream, field, ',')) {
        if (field == "review") {
            review_idx = col_idx;
        } else if (field == "sentiment") {
            sentiment_idx = col_idx;
        }
        col_idx++;
    }

    // check if required columns are present
    if (review_idx == -1 || sentiment_idx == -1) {
        throw std::runtime_error("could not find required columns 'review' and 'sentiment' in csv header");
    }

    std::vector<TrainingExample> examples;
    size_t count = 0;

    // read data rows up to no_examples
    while (std::getline(file, line) && count < no_examples) {
        std::stringstream row_stream(line);
        std::string cell;
        std::string review;
        float sentiment;
        int current_col = 0;

        // go through cells storing review and sentiment based on column indices
        while (std::getline(row_stream, cell, ',')) {
            if (current_col == review_idx) {
                review = cell;
            } else if (current_col == sentiment_idx) {
                try {
                    // try to convert to float
                    sentiment = std::stof(cell);
                } catch (const std::exception& e) {
                    throw std::runtime_error("could not convert sentiment to float: " + cell);
                }
            }
            current_col++;
        }

        // store valid examples
        if (!review.empty()) {
            // create target tensor
            Tensor3D target;
            if (sentiment == 1) {
                target = Tensor3D(1, 2, 1, std::vector<float>{1, 0});
            } else {
                target = Tensor3D(1, 2, 1, std::vector<float>{0, 1});
            }

            // tokenise review and create sequence tensor
            std::vector<Tensor3D> sequence = tokeniser.string_to_embeddings(review);

            // create training example
            TrainingExample example = {sequence, target};
            examples.push_back(example);
            count++;
        }
    }

    return examples;
}

int main() {
    // load embeddings and training data
    // paths are relative to compiled executable location
    Tokeniser tokeniser("../data/glove.6B.100d.txt");
    std::vector<TrainingExample> all_examples = training_examples_from_csv("../data/imdb_clean.csv", tokeniser, 1000);

    // split into training and test data
    size_t size = all_examples.size();
    const float split_ratio = 0.9f;

    std::vector<TrainingExample> training_data =
        std::vector<TrainingExample>(all_examples.begin(), all_examples.begin() + static_cast<size_t>(size * split_ratio));
    std::vector<TrainingExample> test_data =
        std::vector<TrainingExample>(all_examples.begin() + static_cast<size_t>(size * split_ratio), all_examples.end());

    auto input_features = 100;
    size_t hidden_size = 64;
    size_t output_size = 2;
    std::vector<int> mlp_topology = {static_cast<int>(hidden_size), 32, static_cast<int>(output_size)};
    std::vector<std::string> mlp_activation_functions = {"relu", "relu", "softmax"};

    Predictor predictor(input_features, hidden_size, output_size, mlp_topology);
    predictor.set_gru_optimiser(std::make_unique<GRUAdamWOptimiser>());
    predictor.set_mlp_optimiser(std::make_unique<MLPAdamWOptimiser>());
    predictor.set_loss(std::make_unique<CrossEntropyLoss>());

    predictor.train(training_data, test_data, 75, 50);

    // print some example predictions
    std::cout << "\nSample predictions:" << std::endl;
    for (size_t i = 0; i < std::min(size_t(10), test_data.size()); i++) {
        Tensor3D prediction = predictor.predict(test_data[i].sequence);
        std::cout << "Predicted: " << prediction(1, 0, 0) << " Actual: " << test_data[i].target(1, 0, 0) << std::endl;
    }

    return 0;
}