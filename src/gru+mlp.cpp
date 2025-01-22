#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "../include/Tensor3d.hpp"
#include "../include/loss.hpp"
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
    std::vector<Tensor3d> sequence;
    Tensor3d target;

    friend std::ostream& operator<<(std::ostream& os, const TrainingExample& example) {
        os << "sequence length: " << example.sequence.size() << "\n";
        os << "target: \n" << example.target << "\n";
        return os;
    }
};

struct AttentionGradients {
    Tensor3d dL_dW;
    Tensor3d dL_dv;
    size_t attention_size;
    size_t hidden_size;

    AttentionGradients(size_t attention_size, size_t hidden_size)
        : dL_dW(attention_size, hidden_size),
          dL_dv(1, attention_size),
          attention_size(attention_size),
          hidden_size(hidden_size) {}

    // operator overloading for addition
    AttentionGradients operator+(const AttentionGradients& other) const {
        if (attention_size != other.attention_size or hidden_size != other.hidden_size) {
            throw std::invalid_argument("AttentionGradients dimensions don't match for addition");
        }
        AttentionGradients result(attention_size, hidden_size);
        result.dL_dW = dL_dW + other.dL_dW;
        result.dL_dv = dL_dv + other.dL_dv;
        return result;
    }

    // operator overloading for scalar multiplication
    AttentionGradients operator*(float scalar) const {
        AttentionGradients result(attention_size, hidden_size);
        result.dL_dW = dL_dW * scalar;
        result.dL_dv = dL_dv * scalar;
        return result;
    }

    float calculate_norm() const {
        // use l2 norm across all gradients
        float sum_squared = 0.0f;
        size_t total_elements = 0;

        // accumulate squared values from all gradient tensors
        auto accumulate = [&](const Tensor3d& tensor) {
            for (const auto& val : tensor.get_flat_data()) {
                sum_squared += val * val;
                total_elements++;
            }
        };

        // process all gradient tensors
        accumulate(dL_dW);
        accumulate(dL_dv);

        return std::sqrt(sum_squared / total_elements);
    }

    // scale all gradients by a factor
    void scale(float factor) {
        dL_dW = dL_dW * factor;
        dL_dv = dL_dv * factor;
    }
};

struct GRUGradients {
    Tensor3d dW_z, dU_z, db_z;  // update gate gradients
    Tensor3d dW_r, dU_r, db_r;  // reset gate gradients
    Tensor3d dW_h, dU_h, db_h;  // hidden state gradients
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

    float calculate_norm() const {
        // use l2 norm across all gradients
        float sum_squared = 0.0f;
        size_t total_elements = 0;

        // accumulate squared values from all gradient tensors
        auto accumulate = [&](const Tensor3d& tensor) {
            for (const auto& val : tensor.get_flat_data()) {
                sum_squared += val * val;
                total_elements++;
            }
        };

        // process all gradient tensors
        accumulate(dW_z);
        accumulate(dU_z);
        accumulate(db_z);
        accumulate(dW_r);
        accumulate(dU_r);
        accumulate(db_r);
        accumulate(dW_h);
        accumulate(dU_h);
        accumulate(db_h);

        return std::sqrt(sum_squared / total_elements);
    }

    // scale all gradients by a factor
    void scale(float factor) {
        dW_z = dW_z * factor;
        dU_z = dU_z * factor;
        db_z = db_z * factor;
        dW_r = dW_r * factor;
        dU_r = dU_r * factor;
        db_r = db_r * factor;
        dW_h = dW_h * factor;
        dU_h = dU_h * factor;
        db_h = db_h * factor;
    }
};

struct MLPGradients {
    std::vector<std::vector<Tensor3d>> gradients;

    MLPGradients operator+(const MLPGradients& other) const {
        MLPGradients result;
        if (gradients.empty()) {
            result.gradients = other.gradients;
            return result;
        }
        if (other.gradients.empty()) {
            result.gradients = gradients;
            return result;
        }

        // check if the number of layers are the same
        if (gradients.size() != other.gradients.size()) {
            throw std::runtime_error("MLPGradients layers don't match for addition");
        }

        result.gradients.reserve(gradients.size());
        for (size_t i = 0; i < gradients.size(); i++) {
            std::vector<Tensor3d> layer_grads;
            layer_grads.reserve(gradients[i].size());
            for (size_t j = 0; j < gradients[i].size(); j++) {
                layer_grads.push_back(gradients[i][j] + other.gradients[i][j]);
            }
            result.gradients.push_back(layer_grads);
        }
        return result;
    }

    MLPGradients operator*(float scalar) const {
        MLPGradients result;
        result.gradients.reserve(gradients.size());
        for (size_t i = 0; i < gradients.size(); i++) {
            std::vector<Tensor3d> layer_grads;
            layer_grads.reserve(gradients[i].size());
            for (size_t j = 0; j < gradients[i].size(); j++) {
                layer_grads.push_back(gradients[i][j] * scalar);
            }
            result.gradients.push_back(layer_grads);
        }
        return result;
    }

    float calculate_norm() const {
        float sum_squared = 0.0f;
        size_t total_elements = 0;

        for (const auto& layer : gradients) {
            for (const auto& tensor : layer) {
                for (const auto& val : tensor.get_flat_data()) {
                    sum_squared += val * val;
                    total_elements++;
                }
            }
        }

        return std::sqrt(sum_squared / total_elements);
    }

    void scale(float factor) {
        for (auto& layer : gradients) {
            for (auto& tensor : layer) {
                tensor = tensor * factor;
            }
        }
    }
};

class GRUCell {
   private:
    // store sequence of states for BPTT
    struct TimeStep {
        Tensor3d z, r, h_candidate, h;
        Tensor3d h_prev;
        Tensor3d x;

        TimeStep(size_t hidden_size, size_t input_size)
            : z(hidden_size, 1),
              r(hidden_size, 1),
              h_candidate(hidden_size, 1),
              h(hidden_size, 1),
              h_prev(hidden_size, 1),
              x(input_size, 1) {}
    };

    // store the intermediate states for each timestep for each thread
    std::unordered_map<std::thread::id, std::vector<TimeStep>> time_steps;
    std::unique_ptr<std::mutex> timesteps_mutex;

    // clear the stored states
    void clear_states() {
        std::lock_guard<std::mutex> lock(*timesteps_mutex);
        time_steps.erase(std::this_thread::get_id());
    }

    // get the timesteps for the current thread
    std::vector<TimeStep>& get_thread_timesteps() {
        std::lock_guard<std::mutex> lock(*timesteps_mutex);
        return time_steps[std::this_thread::get_id()];
    }

    // store the gstdradients and the gradient of the hidden state from the previous timestep
    struct BackwardResult {
        GRUGradients grads;
        Tensor3d dh_prev;

        BackwardResult(GRUGradients g, Tensor3d h) : grads(g), dh_prev(h) {}
    };

    // compute gradients for a single timestep
    BackwardResult backward(const Tensor3d& delta_h_t, size_t t) {
        if (t >= get_thread_timesteps().size()) {
            throw std::runtime_error("Time step index out of bounds");
        }

        // get the stored states for this timestep
        const TimeStep& step = get_thread_timesteps()[t];

        // initialise gradients for this timestep
        GRUGradients timestep_grads(input_size, hidden_size);

        // debug - clamping delta_h_t as it went to inf
        float max_value = 1000.0f;  // Choose an appropriate maximum value
        Tensor3d clamped_delta_h_t = delta_h_t.apply([max_value](float x) { return std::clamp(x, -max_value, max_value); });

        // 1. Hidden state gradients
        Tensor3d one_matrix(clamped_delta_h_t.height, clamped_delta_h_t.width);
        for (size_t i = 0; i < one_matrix.height; i++) {
            for (size_t j = 0; j < one_matrix.width; j++) {
                one_matrix(0, i, j) = 1.0f;
            }
        }

        Tensor3d dh_tilde = clamped_delta_h_t.hadamard(one_matrix - step.z);

        Tensor3d dz = clamped_delta_h_t.hadamard(step.h_prev - step.h_candidate);

        // 2. Candidate state gradients
        Tensor3d dg = dh_tilde.hadamard(step.h_candidate.apply([](float x) { return 1.0f - x * x; }));  // tanh derivative

        timestep_grads.dW_h = (dg * step.x.transpose());
        timestep_grads.dU_h = dg * (step.r.hadamard(step.h_prev)).transpose();
        timestep_grads.db_h = dg;

        Tensor3d dx_t = timestep_grads.dW_h.transpose() * dg;
        Tensor3d dr = (timestep_grads.dU_h.transpose() * dg).hadamard(step.h_prev);
        Tensor3d dh_prev = (timestep_grads.dU_h.transpose() * dg).hadamard(step.r);

        // 3. Reset gate gradients
        Tensor3d dr_total = dr.hadamard(step.r.apply(sigmoid_derivative));

        timestep_grads.dW_r = dr_total * step.x.transpose();
        timestep_grads.dU_r = dr_total * step.h_prev.transpose();
        timestep_grads.db_r = dr_total;

        dx_t = dx_t + timestep_grads.dW_r.transpose() * dr_total;
        dh_prev = dh_prev + timestep_grads.dU_r.transpose() * dr_total;

        // 4. Update gate gradients
        Tensor3d dz_total = dz.hadamard(step.z.apply(sigmoid_derivative));

        timestep_grads.dW_z = dz_total * step.x.transpose();
        timestep_grads.dU_z = dz_total * step.h_prev.transpose();
        timestep_grads.db_z = dz_total;

        dx_t = dx_t + timestep_grads.dW_z.transpose() * dz_total;
        dh_prev = dh_prev + timestep_grads.dU_z.transpose() * dz_total;

        // 5. Final hidden state contribution
        dh_prev = dh_prev + clamped_delta_h_t.hadamard(step.z);

        // return both values
        return BackwardResult(timestep_grads, dh_prev);
    }

   public:
    // gate weights and biases
    Tensor3d W_z;  // update gate weights for input
    Tensor3d U_z;  // update gate weights for hidden state
    Tensor3d b_z;  // update gate bias

    Tensor3d W_r;  // reset gate weights for input
    Tensor3d U_r;  // reset gate weights for hidden state
    Tensor3d b_r;  // reset gate bias

    Tensor3d W_h;  // candidate hidden state weights for input
    Tensor3d U_h;  // candidate hidden state weights for hidden state
    Tensor3d b_h;  // candidate hidden state bias

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
          b_h(hidden_size, 1),
          timesteps_mutex(std::make_unique<std::mutex>()) {
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
    Tensor3d get_last_hidden_state() {
        if (get_thread_timesteps().empty()) {
            throw std::runtime_error("No hidden state available - run forward pass first");
        }
        return get_thread_timesteps().back().h;
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
    Tensor3d forward_store(const Tensor3d& x, const Tensor3d& h_prev) {
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

        get_thread_timesteps().push_back(step);
        return step.h;
    }

    // forward pass that does not store states - used for evaluation so we dont clog up time_steps
    Tensor3d forward(const Tensor3d& x, const Tensor3d& h_prev) {
        // update gate
        Tensor3d z = (W_z * x + U_z * h_prev + b_z).apply(sigmoid);

        // reset gate
        Tensor3d r = (W_r * x + U_r * h_prev + b_r).apply(sigmoid);

        // candidate hidden state
        Tensor3d h_candidate = (W_h * x + U_h * (r.hadamard(h_prev)) + b_h).apply(std::tanh);

        // final hidden state
        Tensor3d h = z.hadamard(h_prev) + (z.apply([](float x) { return 1.0f - x; }).hadamard(h_candidate));

        return h;
    }

    GRUGradients backpropagate(const Tensor3d& attention_hidden_state_gradients) {
        Tensor3d dh_next = attention_hidden_state_gradients.col(-1);
        GRUGradients accumulated_grads(input_size, hidden_size);
        reset_gradients(accumulated_grads);

        // backpropagate through time
        for (int t = get_thread_timesteps().size() - 1; t >= 0; --t) {
            BackwardResult result = backward(dh_next, t);
            // combination of dL_dh from attention layer and dL_dh from gru
            dh_next = result.dh_prev + attention_hidden_state_gradients.col(t);

            // accumulate gradients for GRU gates
            accumulated_grads = accumulated_grads + result.grads;
        }

        // clear the stored intermediate states computed in the forward pass
        clear_states();
        return accumulated_grads;
    }
};

// layer class representing a single layer in the neural network
class Layer {
   public:
    Tensor3d weights;
    Tensor3d bias;
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
    Tensor3d feedforward(const Tensor3d& input) {
        // compute pre-activation
        Tensor3d z = weights * input + bias;

        // apply activation function
        Tensor3d output(z.height, z.width);
        if (activation_function == "sigmoid") {
            output = z.apply(sigmoid);
        } else if (activation_function == "relu") {
            output = z.apply(relu);
        } else if (activation_function == "softmax") {
            output = z.softmax();
        } else if (activation_function == "none") {
            output = z;  // no activation
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
    std::vector<Tensor3d> feedforward_backprop(const Tensor3d& input) const {
        // compute pre-activation
        Tensor3d z = weights * input + bias;

        // apply activation function
        Tensor3d output(z.height, z.width);
        if (activation_function == "sigmoid") {
            output = z.apply(sigmoid);
        } else if (activation_function == "relu") {
            output = z.apply(relu);
        } else if (activation_function == "softmax") {
            output = z.softmax();
        } else if (activation_function == "none") {
            output = z;  // no activation
        } else {
            throw std::runtime_error("no activation function found for layer");
        }

        return {output, z};
    }
};

class AttentionLayer {
   private:
    // storing necessary values for backprop
    struct ForwardResult {
        // 1. Y = WH                   // pre-activation
        // 2. T = tanh(Y)              // tanh applied element-wise
        // 3. S = vT                   // raw scores
        // 4. A = softmax(S)           // attention weights
        // 5. C = HA^T (check this)      // context vector

        Tensor3d H;
        Tensor3d A;
        Tensor3d S;
        Tensor3d T;
    };

    struct BackwardResult {
        Tensor3d dL_dW;  // gradient of weights
        Tensor3d dL_dv;  // gradient of scoring vector
        Tensor3d dL_dH;  // gradient of hidden states
    };

    // store the intermediate states for each timestep for each thread
    // std::unordered_map<std::thread::id, std::vector<TimeStep>> time_steps;
    // std::unique_ptr<std::mutex> timesteps_mutex;

    std::unordered_map<std::thread::id, ForwardResult> forward_results;
    std::unique_ptr<std::mutex> forward_results_mutex;

    // get the forward results for the current thread
    ForwardResult& get_thread_forward_results() {
        std::lock_guard<std::mutex> lock(*forward_results_mutex);
        return forward_results[std::this_thread::get_id()];
    }

    // clear the stored states
    void clear_states() {
        std::lock_guard<std::mutex> lock(*forward_results_mutex);
        forward_results.erase(std::this_thread::get_id());
    }

   public:
    Tensor3d weights;
    Tensor3d scoring_vector;
    size_t attention_size;
    size_t hidden_size;

    // assumes hidden_states are (1xhidden_sizex1)
    AttentionLayer(size_t hidden_size, size_t attention_size)
        : weights(attention_size, hidden_size),
          scoring_vector(1, attention_size),
          attention_size(attention_size),
          hidden_size(hidden_size),
          forward_results_mutex(std::make_unique<std::mutex>()) {
        weights.xavier_initialise();
        scoring_vector.xavier_initialise();
    }

    // assumes hidden_states are (1xhidden_sizex1)
    Tensor3d forward(const std::vector<Tensor3d>& hidden_states) {
        // check hidden_states are not empty
        if (hidden_states.empty()) {
            throw std::runtime_error("hidden_states must not be empty");
        }
        // check hidden_states are (1xhidden_sizex1)
        if (hidden_states[0].height != weights.width or hidden_states[0].width != 1) {
            throw std::runtime_error("hidden_states must be (1xhidden_sizex1)");
        }

        // concatenate hidden states into a (1xhidden_sizexno_hidden) tensor
        Tensor3d H(1, hidden_states[0].height, hidden_states.size());
        for (size_t i = 0; i < hidden_states.size(); ++i) {
            for (size_t j = 0; j < hidden_states[i].height; ++j) {
                H(0, j, i) = hidden_states[i](0, j, 0);
            }
        }

        // 1. Y = WH                   // pre-activation
        // 2. T = tanh(Y)              // tanh applied element-wise
        // 3. S = vT                   // raw scores
        // 4. A = softmax(S)           // attention weights
        // 5. C = HA^T (check this)      // context vector
        // store A, S, T for backprop

        // compute attention scores
        Tensor3d T = (weights * H).apply(std::tanh);
        Tensor3d S = scoring_vector * T;

        // across width dimension as S is (1, 1, no_hidden)
        Tensor3d A = S.softmax("width");

        Tensor3d C = H * A.transpose();

        // store intermediate results for backprop
        auto& forward_result = get_thread_forward_results();
        forward_result = {H, A, S, T};

        return C;
    }

    // assumes hidden_states are (1xhidden_sizex1)
    BackwardResult backward(const Tensor3d& dL_dC) {
        // break down the forward pass into more managable steps
        // 1. Y = WH                   // pre-activation
        // 2. T = tanh(Y)              // tanh applied element-wise
        // 3. S = vT                   // raw scores
        // 4. A = softmax(S)           // attention weights
        // 5. C = HA^T (check this)      // context vector

        auto tanh_derivative = [](float x) { return 1.0f - std::tanh(x) * std::tanh(x); };

        auto& forward_result = get_thread_forward_results();

        Tensor3d dC_dA = forward_result.H.transpose();  // normal matrix product derivative rule
        Tensor3d dA_dS = forward_result.A.diag() -
                         (forward_result.A.transpose() *
                          forward_result.A);          // apparantly in this case the derivative through softmax is diag(A) - A^TA
        Tensor3d dS_dT = scoring_vector.transpose();  // normal matrix product derivative rule (but remebering v is a row vector)
        // use hadamard with dT/dY
        Tensor3d dT_dY = forward_result.T.apply(tanh_derivative);  // tanh derivative tanh'(x) = sech^2(x) = 1 - tanh^2(x)
        Tensor3d dY_dW = forward_result.H.transpose();             // normal matrix product derivative rule
        Tensor3d dY_dH = weights;

        Tensor3d dL_dA = (dC_dA * dL_dC).transpose();  // this needs to be (1, sequence_length) like A - transpose and order of
                                                       // operations chosen to fix dimensions
        Tensor3d dL_dS =
            dL_dA * dA_dS;  // this needs to be (1, sequence_length) like S - order of operations chosen to fix dimensions
        Tensor3d dL_dT =
            dS_dT *
            dL_dS;  // this needs to be (attention_size, sequence_length) like T - order of operations chosen to fix dimensions
        Tensor3d dL_dY = dL_dT.hadamard(
            dT_dY);  // tanh derivative tanh'(x) = sech^2(x) = 1 - tanh^2(x), why the hadamard? element-wise is weird
        Tensor3d dL_dW = dL_dY * dY_dW;

        // H has two ways to influence C, direcly by C=HA^T, and indirectly through A.
        Tensor3d dL_dH_direct = dL_dC * forward_result.A;  // normal matrix product derivative rule
        Tensor3d dL_dH_indirect =
            (dY_dH.transpose() *
             dL_dY);  // (hidden_size, attention_size) * (attention_size, sequence_length) = (hidden_size, sequence_length)

        Tensor3d dL_dH = dL_dH_direct + dL_dH_indirect;

        Tensor3d dL_dv = dL_dS * forward_result.T.transpose();

        // clear the intermediate forward pass information for this thread
        clear_states();
        return {dL_dW, dL_dv, dL_dH};
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
    virtual void compute_and_apply_updates(std::vector<Layer>& layers, const std::vector<std::vector<Tensor3d>>& gradients) = 0;

    /**
     * @brief Virtual destructor for the MLPOptimiser class.
     */
    virtual ~MLPOptimiser() = default;

    struct GradientResult {
        MLPGradients gradients;         // list of layers, each layer has a list of weight and bias gradient matrices
        Tensor3d input_layer_gradient;  // gradient of the input layer - for more general use as parts of bigger architectures
        Tensor3d output;                // output of the network
    };

    /**
     * @brief Calculates gradients for a single example.
     * @param layers The layers of the neural network.
     * @param input The input matrix.
     * @param target The target matrix.
     * @return A GradientResult struct containing gradients and the output of the network.
     */
    virtual GradientResult calculate_gradient(const std::vector<Layer>& layers, const Tensor3d& input, const Tensor3d& target,
                                              const Loss& loss) {
        // forward pass
        std::vector<Tensor3d> activations = {input};
        std::vector<Tensor3d> preactivations = {input};

        for (const auto& layer : layers) {
            auto results = layer.feedforward_backprop(activations.back());
            activations.push_back(results[0]);
            preactivations.push_back(results[1]);
        }

        // backward pass
        int num_layers = layers.size();
        std::vector<Tensor3d> deltas;
        deltas.reserve(num_layers);

        // output layer error (δ^L = ∇_a C ⊙ σ'(z^L))
        Tensor3d output_delta = loss.derivative(activations.back(), target);
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
            Tensor3d delta = (layers[l + 1].weights.transpose() * deltas.back());
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
        MLPGradients gradients;
        for (int l = 0; l < num_layers; ++l) {
            Tensor3d weight_gradient = deltas[l] * activations[l].transpose();
            gradients.gradients.push_back({weight_gradient, deltas[l]});
        }

        // as we don't treat the input as a layer, we need to return the input layer errors separately
        Tensor3d input_delta = layers[0].weights.transpose() * deltas[0];

        // return a GradientResult struct for purposes of tracking loss
        return {gradients, input_delta, activations.back()};
    }

    // return clipped gradients to prevent exploding gradients
    MLPGradients clip_gradients(const MLPGradients& gradients, float max_norm = 1.0f) {
        MLPGradients scaled_gradients = gradients;
        float norm = scaled_gradients.calculate_norm();
        if (norm > max_norm) {
            scaled_gradients.scale(max_norm / norm);
            // debug
            std::cout << "clipped mlp\n";
        }
        return scaled_gradients;
    }
};

class MLPSGDOptimiser : public MLPOptimiser {
   private:
    float learning_rate;
    std::vector<std::vector<Tensor3d>> velocity;
    float clip_norm;

   public:
    /**
     * @brief Constructs an MLPSGDOptimiser object with the specified learning rate.
     * @param lr The learning rate (default: 0.1f).
     */
    MLPSGDOptimiser(float lr = 0.1f, float clip_norm = 1.0f) : learning_rate(lr), clip_norm(clip_norm) {}

    /**
     * @brief Initializes the velocity vectors for SGD optimization.
     * @param layers The layers of the neural network.
     */
    void initialise_velocity(const std::vector<Layer>& layers) {
        velocity.clear();
        for (const auto& layer : layers) {
            velocity.push_back(
                {Tensor3d(layer.weights.height, layer.weights.width), Tensor3d(layer.bias.height, layer.bias.width)});
        }
    }

    /**
     * @brief Computes and applies updates using Stochastic Gradient Descent.
     * @param layers The layers of the neural network to update.
     * @param gradients The gradients used for updating the layers.
     */
    void compute_and_apply_updates(std::vector<Layer>& layers, const std::vector<std::vector<Tensor3d>>& gradients) override {
        if (velocity.empty()) {
            initialise_velocity(layers);
        }

        // convert vector of vectors to MLPGradients
        MLPGradients gradients_copy;
        gradients_copy.gradients = gradients;

        // clip gradients
        MLPGradients clipped_gradients = clip_gradients(gradients_copy, clip_norm);

        // compute and apply updates
        for (size_t l = 0; l < layers.size(); ++l) {
            for (int i = 0; i < 2; ++i) {  // 0 for weights, 1 for biases
                // compute adjustment
                velocity[l][i] = clipped_gradients.gradients[l][i] * -learning_rate;
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
    std::vector<std::vector<Tensor3d>> velocity;
    float clip_norm;

   public:
    /**
     * @brief Constructs an MLPSGDMomentumOptimiser object with the specified learning rate and momentum.
     * @param lr The learning rate (default: 0.1f).
     * @param mom The momentum coeficient (default: 0.9f).
     */
    MLPSGDMomentumOptimiser(float lr = 0.1f, float mom = 0.9f, float clip_norm = 1.0f)
        : learning_rate(lr), momentum(mom), clip_norm(clip_norm) {}

    /**
     * @brief Initializes the velocity vectors for SGD with Momentum optimization.
     * @param layers The layers of the neural network.
     */
    void initialise_velocity(const std::vector<Layer>& layers) {
        velocity.clear();
        for (const auto& layer : layers) {
            velocity.push_back(
                {Tensor3d(layer.weights.height, layer.weights.width), Tensor3d(layer.bias.height, layer.bias.width)});
        }
    }

    /**
     * @brief Computes and applies updates using Stochastic Gradient Descent with Momentum.
     * @param layers The layers of the neural network to update.
     * @param gradients The gradients used for updating the layers.
     */
    void compute_and_apply_updates(std::vector<Layer>& layers, const std::vector<std::vector<Tensor3d>>& gradients) override {
        // initialise velocity if needed
        if (velocity.empty()) {
            initialise_velocity(layers);
        }

        // convert vector of vectors to MLPGradients
        MLPGradients gradients_copy;
        gradients_copy.gradients = gradients;

        // clip gradients
        MLPGradients clipped_gradients = clip_gradients(gradients_copy, clip_norm);

        // compute updates
        for (size_t l = 0; l < layers.size(); ++l) {
            for (int i = 0; i < 2; ++i) {  // 0 for weights, 1 for biases
                // compute adjustments
                velocity[l][i] = (velocity[l][i] * momentum) - (clipped_gradients.gradients[l][i] * learning_rate);
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
    std::vector<std::vector<Tensor3d>> m;  // first moment
    std::vector<std::vector<Tensor3d>> v;  // second moment
    float clip_norm;

   public:
    /**
     * @brief Constructs an MLPAdamOptimiser object with the specified parameters.
     * @param lr The learning rate (default: 0.0f01).
     * @param b1 The beta1 parameter (default: 0.9f).
     * @param b2 The beta2 parameter (default: 0.9f99).
     * @param eps The epsilon parameter for numerical stability (default: 1e-8).
     */
    MLPAdamOptimiser(float lr = 0.001f, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-6f, float clip_norm = 1.0f)
        : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), t(0), clip_norm(clip_norm) {}

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
            m.push_back({Tensor3d(layer.weights.height, layer.weights.width), Tensor3d(layer.bias.height, layer.bias.width)});
            v.push_back({Tensor3d(layer.weights.height, layer.weights.width), Tensor3d(layer.bias.height, layer.bias.width)});
        }
    }

    /**
     * @brief Computes and applies updates using the Adam optimization algorithm.
     * @param layers The layers of the neural network to update.
     * @param gradients The gradients used for updating the layers.
     */
    void compute_and_apply_updates(std::vector<Layer>& layers, const std::vector<std::vector<Tensor3d>>& gradients) override {
        if (m.empty() or v.empty()) {
            initialise_moments(layers);
        }

        // convert vector of vectors to MLPGradients
        MLPGradients gradients_copy;
        gradients_copy.gradients = gradients;

        // clip gradients
        MLPGradients clipped_gradients = clip_gradients(gradients_copy, clip_norm);

        t++;  // increment timestep

        for (size_t l = 0; l < layers.size(); ++l) {
            for (int i = 0; i < 2; ++i) {  // 0 for weights, 1 for biases
                // update biased first moment estimate
                m[l][i] = m[l][i] * beta1 + clipped_gradients.gradients[l][i] * (1.0f - beta1);

                // update biased second raw moment estimate
                v[l][i] = v[l][i] * beta2 +
                          clipped_gradients.gradients[l][i].hadamard(clipped_gradients.gradients[l][i]) * (1.0f - beta2);

                // compute bias-corrected first moment estimate
                Tensor3d m_hat = m[l][i] * (1.0f / (1.0f - std::pow(beta1, t)));

                // compute bias-corrected second raw moment estimate
                Tensor3d v_hat = v[l][i] * (1.0f / (1.0f - std::pow(beta2, t)));

                // compute the update
                Tensor3d update = m_hat.hadamard(v_hat.apply([this](float x) { return 1.0f / (std::sqrt(x) + epsilon); }));

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
    std::vector<std::vector<Tensor3d>> m;  // first moment
    std::vector<std::vector<Tensor3d>> v;  // second moment
    float clip_norm;

   public:
    /**
     * @brief Constructs an MLPAdamWOptimiser object with the specified parameters.
     * @param lr The learning rate (default: 0.0f01).
     * @param b1 The beta1 parameter (default: 0.9f).
     * @param b2 The beta2 parameter (default: 0.9f99).
     * @param eps The epsilon parameter for numerical stability (default: 1e-8).
     * @param wd The weight decay parameter (default: 0.0f1).
     */
    MLPAdamWOptimiser(float lr = 0.001f, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-6f, float wd = 0.001f,
                      float clip_norm = 1.0f)
        : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), weight_decay(wd), t(0), clip_norm(clip_norm) {}

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
            m.push_back({Tensor3d(layer.weights.height, layer.weights.width), Tensor3d(layer.bias.height, layer.bias.width)});
            v.push_back({Tensor3d(layer.weights.height, layer.weights.width), Tensor3d(layer.bias.height, layer.bias.width)});
        }
    }

    /**
     * @brief Computes and applies updates using the AdamW optimization algorithm.
     * @param layers The layers of the neural network to update.
     * @param gradients The gradients used for updating the layers.
     */
    void compute_and_apply_updates(std::vector<Layer>& layers, const std::vector<std::vector<Tensor3d>>& gradients) override {
        if (m.empty() || v.empty()) {
            initialise_moments(layers);
        }

        // convert vector of vectors to MLPGradients
        MLPGradients gradients_copy;
        gradients_copy.gradients = gradients;

        // clip gradients
        MLPGradients clipped_gradients = clip_gradients(gradients_copy, clip_norm);

        t++;  // increment timestep

        for (size_t l = 0; l < layers.size(); ++l) {
            for (int i = 0; i < 2; ++i) {  // 0 for weights, 1 for biases
                // update biased first moment estimate
                m[l][i] = m[l][i] * beta1 + clipped_gradients.gradients[l][i] * (1.0f - beta1);

                // update biased second raw moment estimate
                v[l][i] = v[l][i] * beta2 +
                          clipped_gradients.gradients[l][i].hadamard(clipped_gradients.gradients[l][i]) * (1.0f - beta2);

                // compute bias-corrected first moment estimate
                Tensor3d m_hat = m[l][i] * (1.0f / (1.0f - std::pow(beta1, t)));

                // compute bias-corrected second raw moment estimate
                Tensor3d v_hat = v[l][i] * (1.0f / (1.0f - std::pow(beta2, t)));

                // compute the Adam update
                Tensor3d update = m_hat.hadamard(v_hat.apply([this](float x) { return 1.0f / (std::sqrt(x) + epsilon); }));

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

    // return clipped gradients to prevent exploding gradients
    GRUGradients clip_gradients(const GRUGradients& gradients, float max_norm = 1.0f) {
        GRUGradients scaled_gradients = gradients;
        float norm = scaled_gradients.calculate_norm();
        if (norm > max_norm) {
            scaled_gradients.scale(max_norm / norm);
            // debug
            std::cout << "clipped gru\n";
        }

        return scaled_gradients;
    }
};

class GRUSGDOptimiser : public GRUOptimiser {
   private:
    float learning_rate;
    float clip_norm;

   public:
    GRUSGDOptimiser(float lr = 0.1f, float clip_norm = 1.0f) : learning_rate(lr), clip_norm(clip_norm) {}

    void compute_and_apply_updates(GRUCell& gru, const GRUGradients& grads) override {
        // clip grads
        GRUGradients clipped_gradients = clip_gradients(grads, clip_norm);

        // update weights and biases using gradients
        gru.W_z = gru.W_z - clipped_gradients.dW_z * learning_rate;
        gru.U_z = gru.U_z - clipped_gradients.dU_z * learning_rate;
        gru.b_z = gru.b_z - clipped_gradients.db_z * learning_rate;

        gru.W_r = gru.W_r - clipped_gradients.dW_r * learning_rate;
        gru.U_r = gru.U_r - clipped_gradients.dU_r * learning_rate;
        gru.b_r = gru.b_r - clipped_gradients.db_r * learning_rate;

        gru.W_h = gru.W_h - clipped_gradients.dW_h * learning_rate;
        gru.U_h = gru.U_h - clipped_gradients.dU_h * learning_rate;
        gru.b_h = gru.b_h - clipped_gradients.db_h * learning_rate;
    }
};

class GRUSGDMomentumOptimiser : public GRUOptimiser {
   private:
    float learning_rate;
    float momentum;
    GRUGradients velocity;
    float clip_norm;

   public:
    GRUSGDMomentumOptimiser(float lr = 0.1f, float mom = 0.9f, float clip_norm = 1.0f)
        : learning_rate(lr), momentum(mom), velocity(0, 0), clip_norm(clip_norm) {}  // sizes will be set on first use

    void compute_and_apply_updates(GRUCell& gru, const GRUGradients& grads) override {
        // initialise velocity if needed
        if (velocity.dW_z.height == 0) {
            velocity = GRUGradients(gru.input_size, gru.hidden_size);
        }

        // clip grads
        GRUGradients clipped_gradients = clip_gradients(grads, clip_norm);

        // update velocities and apply updates for each parameter
        velocity.dW_z = velocity.dW_z * momentum - clipped_gradients.dW_z * learning_rate;
        velocity.dU_z = velocity.dU_z * momentum - clipped_gradients.dU_z * learning_rate;
        velocity.db_z = velocity.db_z * momentum - clipped_gradients.db_z * learning_rate;

        velocity.dW_r = velocity.dW_r * momentum - clipped_gradients.dW_r * learning_rate;
        velocity.dU_r = velocity.dU_r * momentum - clipped_gradients.dU_r * learning_rate;
        velocity.db_r = velocity.db_r * momentum - clipped_gradients.db_r * learning_rate;

        velocity.dW_h = velocity.dW_h * momentum - clipped_gradients.dW_h * learning_rate;
        velocity.dU_h = velocity.dU_h * momentum - clipped_gradients.dU_h * learning_rate;
        velocity.db_h = velocity.db_h * momentum - clipped_gradients.db_h * learning_rate;

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
    float clip_norm;

    void update_parameter(Tensor3d& param, Tensor3d& m_param, Tensor3d& v_param, const Tensor3d& grad) {
        // Update biased first moment estimate
        m_param = m_param * beta1 + grad * (1.0f - beta1);

        // Update biased second raw moment estimate
        v_param = v_param * beta2 + grad.hadamard(grad) * (1.0f - beta2);

        // Compute bias-corrected first moment estimate
        Tensor3d m_hat = m_param * (1.0f / (1.0f - std::pow(beta1, t)));

        // Compute bias-corrected second raw moment estimate
        Tensor3d v_hat = v_param * (1.0f / (1.0f - std::pow(beta2, t)));

        // Update parameters
        Tensor3d denom = v_hat.apply([this](float x) { return 1.0f / (std::sqrt(x) + epsilon); });
        Tensor3d update = m_hat.hadamard(denom);
        param = param - update * learning_rate;
    }

   public:
    GRUAdamOptimiser(float lr = 0.001f, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-6f, float clip_norm = 1.0f)
        : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), t(0), m(0, 0), v(0, 0), clip_norm(clip_norm) {}

    void compute_and_apply_updates(GRUCell& gru, const GRUGradients& grads) override {
        if (m.dW_z.height == 0) {
            m = GRUGradients(gru.input_size, gru.hidden_size);
            v = GRUGradients(gru.input_size, gru.hidden_size);
        }

        // clip grads
        GRUGradients clipped_gradients = clip_gradients(grads, clip_norm);

        t++;

        // Update all parameters
        update_parameter(gru.W_z, m.dW_z, v.dW_z, clipped_gradients.dW_z);
        update_parameter(gru.U_z, m.dU_z, v.dU_z, clipped_gradients.dU_z);
        update_parameter(gru.b_z, m.db_z, v.db_z, clipped_gradients.db_z);
        update_parameter(gru.W_r, m.dW_r, v.dW_r, clipped_gradients.dW_r);
        update_parameter(gru.U_r, m.dU_r, v.dU_r, clipped_gradients.dU_r);
        update_parameter(gru.b_r, m.db_r, v.db_r, clipped_gradients.db_r);
        update_parameter(gru.W_h, m.dW_h, v.dW_h, clipped_gradients.dW_h);
        update_parameter(gru.U_h, m.dU_h, v.dU_h, clipped_gradients.dU_h);
        update_parameter(gru.b_h, m.db_h, v.db_h, clipped_gradients.db_h);
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
    float clip_norm;

    void update_parameter(Tensor3d& param, Tensor3d& m_param, Tensor3d& v_param, const Tensor3d& grad,
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
        Tensor3d m_hat = m_param * (1.0f / (1.0f - std::pow(beta1, t)));

        // Compute bias-corrected second raw moment estimate
        Tensor3d v_hat = v_param * (1.0f / (1.0f - std::pow(beta2, t)));

        // Update parameters
        Tensor3d denom = v_hat.apply([this](float x) { return 1.0f / (std::sqrt(x) + epsilon); });
        Tensor3d update = m_hat.hadamard(denom);
        param = param - update * learning_rate;
    }

   public:
    GRUAdamWOptimiser(float lr = 0.001f, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-6f, float wd = 0.001f,
                      float clip_norm = 1.0f)
        : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), weight_decay(wd), t(0), m(0, 0), v(0, 0), clip_norm(clip_norm) {}

    void compute_and_apply_updates(GRUCell& gru, const GRUGradients& grads) override {
        // initialise velocity/momentums if needed
        if (m.dW_z.height == 0) {
            m = GRUGradients(gru.input_size, gru.hidden_size);
            v = GRUGradients(gru.input_size, gru.hidden_size);
        }

        // clip grads
        GRUGradients clipped_gradients = clip_gradients(grads, clip_norm);

        t++;

        // Update weights (with weight decay)
        update_parameter(gru.W_z, m.dW_z, v.dW_z, clipped_gradients.dW_z, true);
        update_parameter(gru.U_z, m.dU_z, v.dU_z, clipped_gradients.dU_z, true);
        update_parameter(gru.W_r, m.dW_r, v.dW_r, clipped_gradients.dW_r, true);
        update_parameter(gru.U_r, m.dU_r, v.dU_r, clipped_gradients.dU_r, true);
        update_parameter(gru.W_h, m.dW_h, v.dW_h, clipped_gradients.dW_h, true);
        update_parameter(gru.U_h, m.dU_h, v.dU_h, clipped_gradients.dU_h, true);

        // Update biases (without weight decay)
        update_parameter(gru.b_z, m.db_z, v.db_z, clipped_gradients.db_z, false);
        update_parameter(gru.b_r, m.db_r, v.db_r, clipped_gradients.db_r, false);
        update_parameter(gru.b_h, m.db_h, v.db_h, clipped_gradients.db_h, false);
    }
};

class AttentionOptimiser {
   public:
    virtual void compute_and_apply_updates(AttentionLayer& attention, const AttentionGradients& gradients) = 0;
    virtual ~AttentionOptimiser() = default;

    AttentionGradients clip_gradients(const AttentionGradients& gradients, float max_norm = 1.0f) {
        AttentionGradients scaled_gradients = gradients;
        float norm = scaled_gradients.calculate_norm();
        if (norm > max_norm) {
            scaled_gradients.scale(max_norm / norm);
            // debug
            std::cout << "clipped attention\n";
        }

        return scaled_gradients;
    }
};

class AttentionAdamWOptimiser : public AttentionOptimiser {
   private:
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    float weight_decay;
    int t;
    AttentionGradients m;
    AttentionGradients v;
    float clip_norm;

    void update_parameter(Tensor3d& param, Tensor3d& m_param, Tensor3d& v_param, const Tensor3d& grad,
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
        Tensor3d m_hat = m_param * (1.0f / (1.0f - std::pow(beta1, t)));

        // Compute bias-corrected second raw moment estimate
        Tensor3d v_hat = v_param * (1.0f / (1.0f - std::pow(beta2, t)));

        // Update parameters
        Tensor3d denom = v_hat.apply([this](float x) { return 1.0f / (std::sqrt(x) + epsilon); });
        Tensor3d update = m_hat.hadamard(denom);
        param = param - update * learning_rate;
    }

   public:
    AttentionAdamWOptimiser(float lr = 0.001f, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-6f, float wd = 0.001f,
                            float clip_norm = 1.0f)
        : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), weight_decay(wd), t(0), m(0, 0), v(0, 0), clip_norm(clip_norm) {}

    void compute_and_apply_updates(AttentionLayer& attention, const AttentionGradients& gradients) override {
        // intialise m and v if needed
        if (m.dL_dW.height == 0) {
            m = AttentionGradients(attention.attention_size, attention.hidden_size);
            v = AttentionGradients(attention.attention_size, attention.hidden_size);
        }

        // clip gradients
        AttentionGradients clipped_gradients = clip_gradients(gradients, clip_norm);

        t++;

        // update weights and biases using gradients
        update_parameter(attention.weights, m.dL_dW, v.dL_dW, clipped_gradients.dL_dW, true);
        update_parameter(attention.scoring_vector, m.dL_dv, v.dL_dv, clipped_gradients.dL_dv, true);
    }
};

// -----------------------------------------------------------------------------------------------------

// load only batches of examples from csv file at a time - avoids lack of memory issues
class BatchDataLoader {
   public:
    size_t batch_size;
    size_t no_examples;

   private:
    std::string filename;
    Tokeniser& tokeniser;
    std::ifstream file;
    std::string header;
    int review_idx;
    int sentiment_idx;

    // store the byte offset of each line in the file
    std::vector<std::streampos> line_positions;
    std::vector<size_t> indices;
    size_t current_index;

    // initialise line positions - only done once when loader is created
    void init_line_positions() {
        std::string line;

        // store header position
        std::streampos header_pos = file.tellg();
        std::getline(file, line);

        // store all other line positions
        while (file.good()) {
            std::streampos pos = file.tellg();
            std::getline(file, line);
            if (!line.empty()) {
                line_positions.push_back(pos);
            }
        }

        // prepare indices for shuffling
        indices.resize(line_positions.size());
        std::iota(indices.begin(), indices.end(), 0);

        no_examples = indices.size();
    }

   public:
    BatchDataLoader(const std::string& filename, Tokeniser& tokeniser, size_t batch_size)
        : filename(filename), tokeniser(tokeniser), batch_size(batch_size), file(filename), current_index(0) {
        if (!file.is_open()) {
            throw std::runtime_error("could not open file: " + filename);
        }

        // read and process header
        std::getline(file, header);
        std::stringstream header_stream(header);
        std::string field;
        review_idx = -1;
        sentiment_idx = -1;
        int col_idx = 0;

        while (std::getline(header_stream, field, ',')) {
            if (field == "review") {
                review_idx = col_idx;
            } else if (field == "sentiment") {
                sentiment_idx = col_idx;
            }
            col_idx++;
        }

        if (review_idx == -1 || sentiment_idx == -1) {
            throw std::runtime_error("could not find required columns 'review' and 'sentiment' in csv header");
        }

        // initialise line positions and indices
        init_line_positions();
    }

    // load next batch of examples
    std::vector<TrainingExample> next_batch(int no_examples = -1) {
        // by default, load a full batch
        if (no_examples == -1) {
            no_examples = batch_size;
        }

        std::vector<TrainingExample> batch;

        if (current_index >= indices.size()) {
            return batch;  // return empty batch if we've processed all data - use to trigger end of epoch
        }

        size_t examples_to_process = std::min(batch_size, indices.size() - current_index);

        for (size_t i = 0; i < examples_to_process; ++i) {
            // jump directly to the line we want using stored position
            file.clear();  // clear any error flags
            file.seekg(line_positions[indices[current_index + i]]);

            std::string line;
            std::getline(file, line);

            std::stringstream row_stream(line);
            std::string cell;
            std::string review;
            float sentiment;
            int current_col = 0;

            while (std::getline(row_stream, cell, ',')) {
                if (current_col == review_idx) {
                    review = cell;
                } else if (current_col == sentiment_idx) {
                    try {
                        sentiment = std::stof(cell);
                    } catch (const std::exception& e) {
                        throw std::runtime_error("could not convert sentiment to float: " + cell);
                    }
                }
                current_col++;
            }

            if (!review.empty()) {
                Tensor3d target;
                if (sentiment == 1) {
                    target = Tensor3d(1, 2, 1, std::vector<float>{1, 0});
                } else {
                    target = Tensor3d(1, 2, 1, std::vector<float>{0, 1});
                }

                std::vector<Tensor3d> sequence = tokeniser.string_to_embeddings(review);
                batch.push_back({sequence, target});
            }
        }

        current_index += examples_to_process;
        return batch;
    }

    void reset() {
        current_index = 0;
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);
    }
};

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
    Tensor3d feedforward(const Tensor3d& input) {
        Tensor3d current = input;
        for (auto& layer : layers) {
            current = layer.feedforward(current);
        }
        return current;
    }

    size_t get_index_of_max_element_in_nx1_matrix(const Tensor3d& matrix) const {
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
    AttentionLayer attention;
    MLP mlp;

    size_t input_size;
    size_t hidden_size;
    size_t attention_size;
    size_t output_size;

    std::unique_ptr<GRUOptimiser> gru_optimiser;
    std::unique_ptr<AttentionOptimiser> attention_optimiser;
    std::unique_ptr<MLPOptimiser> mlp_optimiser;
    std::unique_ptr<Loss> loss;

   public:
    Predictor(size_t input_size, size_t hidden_size, size_t attention_size, size_t output_size, std::vector<int> mlp_topology,
              std::vector<std::string> mlp_activation_functions = {})
        : gru(input_size, hidden_size),
          attention(hidden_size, attention_size),
          mlp(mlp_topology, mlp_activation_functions),
          input_size(input_size),
          hidden_size(hidden_size),
          attention_size(attention_size),
          output_size(output_size) {}

    // set optimiser - call before training
    void set_gru_optimiser(std::unique_ptr<GRUOptimiser> new_optimiser) { gru_optimiser = std::move(new_optimiser); }
    // set optimiser - call before training
    void set_attention_optimiser(std::unique_ptr<AttentionOptimiser> new_optimiser) {
        attention_optimiser = std::move(new_optimiser);
    }
    // set optimiser - call before training
    void set_mlp_optimiser(std::unique_ptr<MLPOptimiser> new_optimiser) { mlp_optimiser = std::move(new_optimiser); }
    // set loss function - call before training
    void set_loss(std::unique_ptr<Loss> new_loss) { loss = std::move(new_loss); }

    // update parameters using optimisers
    void update_parameters(const GRUGradients& gru_grads, const MLPGradients& mlp_grads,
                           const AttentionGradients& attention_grads) {
        if (!gru_optimiser) {
            throw std::runtime_error("no optimiser set");
        }
        gru_optimiser->compute_and_apply_updates(gru, gru_grads);
        mlp_optimiser->compute_and_apply_updates(mlp.layers, mlp_grads.gradients);
        attention_optimiser->compute_and_apply_updates(attention, attention_grads);
    }

    // process sequence and return prediction (full feedforward pass)
    Tensor3d predict(const std::vector<Tensor3d>& input_sequence, bool store_states = false) {
        if (input_sequence.empty()) {
            throw std::runtime_error("can't predict on empty input sequence");
        }

        // initialise hidden state
        Tensor3d h_t(hidden_size, 1);

        // process sequence through GRU
        std::vector<Tensor3d> h_states;
        for (const auto& x : input_sequence) {
            if (store_states) {
                h_t = gru.forward_store(x, h_t);
            } else {
                h_t = gru.forward(x, h_t);
            }
            h_states.push_back(h_t);
        }

        // process through attention layer
        h_t = attention.forward(h_states);

        // final linear layer
        return mlp.feedforward(h_t);
    }

    // process sequence and return hidden states (used for backpropagation)
    std::vector<Tensor3d> feedforward_gru(const std::vector<Tensor3d>& input_sequence) {
        std::vector<Tensor3d> h_states;
        Tensor3d h_t(hidden_size, 1);

        // process sequence through GRU
        for (const auto& x : input_sequence) {
            h_t = gru.forward_store(x, h_t);
            h_states.push_back(h_t);
        }

        // final hidden state
        return h_states;
    }

    // gets the gradients for a single training example
    std::tuple<GRUGradients, AttentionGradients, MLPGradients> compute_gradients(const std::vector<Tensor3d>& input_sequence,
                                                                                 const Tensor3d& target) {
        // forward pass
        Tensor3d context_vector = attention.forward(feedforward_gru(input_sequence));

        auto [mlp_gradients, input_layer_gradient, output] =
            mlp_optimiser->calculate_gradient(mlp.layers, context_vector, target, *loss);

        // backpropagate through attention layer
        auto [dL_dW, dL_dv, dL_dH] = attention.backward(input_layer_gradient);

        // create attention gradients struct to return
        AttentionGradients attention_gradients(dL_dW.height, dL_dv.width);
        attention_gradients.dL_dW = dL_dW;
        attention_gradients.dL_dv = dL_dv;

        // backpropagate through GRU
        auto gru_gradients = gru.backpropagate(dL_dH);
        return {gru_gradients, attention_gradients, mlp_gradients};
    }

    struct GradientMagnitudes {
        // gru magnitudes
        float gru_weights_max;
        float gru_weights_avg;
        // mlp magnitudes
        float mlp_weights_max;
        float mlp_weights_avg;

        friend std::ostream& operator<<(std::ostream& os, const GradientMagnitudes& m) {
            os << "GRU - max: " << m.gru_weights_max << ", avg: " << m.gru_weights_avg << "\n";
            os << "MLP - max: " << m.mlp_weights_max << ", avg: " << m.mlp_weights_avg;
            return os;
        }
    };

    struct WeightMagnitudes {
        float gru_weights_max;
        float gru_weights_avg;
        float mlp_weights_max;
        float mlp_weights_avg;

        friend std::ostream& operator<<(std::ostream& os, const WeightMagnitudes& m) {
            os << "GRU weights - max: " << m.gru_weights_max << ", avg: " << m.gru_weights_avg
               << " | MLP weights - max: " << m.mlp_weights_max << ", avg: " << m.mlp_weights_avg;
            return os;
        }
    };

    WeightMagnitudes get_weight_magnitudes() const {
        WeightMagnitudes magnitudes{0.0f, 0.0f, 0.0f, 0.0f};
        float gru_sum = 0.0f;
        int gru_count = 0;

        // process gru weights
        auto process_tensor = [&](const Tensor3d& t) {
            auto [max_val, avg_val] = t.get_magnitudes();
            magnitudes.gru_weights_max = std::max(magnitudes.gru_weights_max, max_val);
            gru_sum += avg_val * t.get_flat_data().size();
            gru_count += t.get_flat_data().size();
        };

        // check all gru weights
        process_tensor(gru.W_z);
        process_tensor(gru.U_z);
        process_tensor(gru.b_z);
        process_tensor(gru.W_r);
        process_tensor(gru.U_r);
        process_tensor(gru.b_r);
        process_tensor(gru.W_h);
        process_tensor(gru.U_h);
        process_tensor(gru.b_h);

        magnitudes.gru_weights_avg = gru_sum / gru_count;

        // process mlp weights
        float mlp_sum = 0.0f;
        int mlp_count = 0;

        for (const auto& layer : mlp.layers) {
            auto [max_val, avg_val] = layer.weights.get_magnitudes();
            magnitudes.mlp_weights_max = std::max(magnitudes.mlp_weights_max, max_val);
            mlp_sum += avg_val * layer.weights.get_flat_data().size();
            mlp_count += layer.weights.get_flat_data().size();
        }

        magnitudes.mlp_weights_avg = mlp_sum / mlp_count;
        return magnitudes;
    }

    void train_with_batches(BatchDataLoader& train_loader, BatchDataLoader& test_loader, int epochs,
                            size_t no_threads = std::thread::hardware_concurrency()) {
        if (!gru_optimiser or !mlp_optimiser or !loss or !attention_optimiser) {
            throw std::runtime_error("optimisers or loss function not set");
        }

        // check if number of no_threads is valid
        const int max_threads = std::thread::hardware_concurrency();
        if (no_threads > max_threads) {
            throw std::runtime_error("number of threads cannot be greater than the number of hardware threads");
        }

        size_t full_batches = train_loader.no_examples / train_loader.batch_size;

        for (int epoch = 0; epoch < epochs; epoch++) {
            int batch_count = 0;

            std::cout << "Starting epoch " << epoch + 1 << "/" << epochs << std::endl;

            while (true) {
                auto batch = train_loader.next_batch();
                if (batch.empty()) break;  // end of epoch

                // divide batch into mini-batches for parallel processing
                size_t mini_batch_size = (batch.size() + no_threads - 1) / no_threads;

                // initialise storage for each threads calculated gradients
                std::vector<GRUGradients> thread_gru_grads(no_threads, GRUGradients(input_size, hidden_size));
                std::vector<AttentionGradients> thread_attention_grads(
                    no_threads, AttentionGradients(attention.attention_size, attention.hidden_size));
                std::vector<MLPGradients> thread_mlp_grads(no_threads);
                std::vector<std::thread> threads;

                // launch threads
                for (int t = 0; t < no_threads; t++) {
                    threads.emplace_back([&, t]() {
                        size_t start = t * mini_batch_size;
                        size_t end = std::min(start + mini_batch_size, batch.size());

                        // process mini-batch
                        GRUGradients local_gru_grads(input_size, hidden_size);
                        AttentionGradients local_attention_grads(attention.attention_size, attention.hidden_size);
                        MLPGradients local_mlp_grads;

                        for (size_t i = start; i < end; i++) {
                            // calculate and accumulate gradients for the i'th example in the mini-batch
                            auto [gru_grads, attention_grads, mlp_grads] = compute_gradients(batch[i].sequence, batch[i].target);
                            local_gru_grads = local_gru_grads + gru_grads;
                            local_attention_grads = local_attention_grads + attention_grads;
                            local_mlp_grads = local_mlp_grads + mlp_grads;
                        }

                        // store thread results
                        thread_gru_grads[t] = local_gru_grads;
                        thread_mlp_grads[t] = local_mlp_grads;
                        thread_attention_grads[t] = local_attention_grads;
                    });
                }

                // wait for all threads to complete
                for (auto& thread : threads) {
                    thread.join();
                }

                // combine gradients from all threads
                GRUGradients averaged_gru_gradients(input_size, hidden_size);
                MLPGradients averaged_mlp_gradients;
                AttentionGradients averaged_attention_gradients(attention.attention_size, attention.hidden_size);

                for (int t = 0; t < no_threads; t++) {
                    averaged_gru_gradients = averaged_gru_gradients + thread_gru_grads[t];
                    averaged_mlp_gradients = averaged_mlp_gradients + thread_mlp_grads[t];
                    averaged_attention_gradients = averaged_attention_gradients + thread_attention_grads[t];
                }

                // average gradients
                averaged_gru_gradients = averaged_gru_gradients * (1.0f / batch.size());
                averaged_mlp_gradients = averaged_mlp_gradients * (1.0f / batch.size());
                averaged_attention_gradients = averaged_attention_gradients * (1.0f / batch.size());

                // update parameters
                update_parameters(averaged_gru_gradients, averaged_mlp_gradients, averaged_attention_gradients);

                batch_count++;

                if (batch_count % 10 == 0) {
                    std::cout << "\rBatch " << batch_count << "/" << full_batches << " complete";
                    // evaluate on test set
                    float total_accuracy = 0.0f;
                    int test_batches = 0;
                    test_loader.reset();  // reset test loader to beginning

                    auto test_batch = test_loader.next_batch(200);
                    auto metrics = evaluate(test_batch);
                    std::cout << " - rough test accuracy: " << metrics.accuracy * 100.0f << "%";

                    // get weight magnitudes
                    auto magnitudes = get_weight_magnitudes();
                    std::cout << " - " << magnitudes << std::endl;

                } else {
                    std::cout << "\rBatch " << batch_count << "/" << full_batches + 1 << " complete";
                    // get weight magnitudes
                    auto magnitudes = get_weight_magnitudes();
                    std::cout << " - " << magnitudes << std::endl;
                }
            }

            std::cout << "\rEpoch " << epoch + 1 << "/" << epochs << " complete    " << std::endl;
            std::string model_path = "../models/model_" + std::to_string(epoch) + ".bin";
            save_model(model_path);

            // evaluate on test set
            float total_accuracy = 0.0f;
            int test_batches = 0;
            test_loader.reset();  // reset test loader to beginning

            while (true) {
                auto test_batch = test_loader.next_batch();
                if (test_batch.empty()) break;

                auto metrics = evaluate(test_batch);
                total_accuracy += metrics.accuracy;
                test_batches++;
            }

            float avg_accuracy = total_accuracy / test_batches;
            std::cout << " - full test accuracy: " << avg_accuracy * 100.0f << "%\n" << std::endl;

            // reset train loader for next epoch
            train_loader.reset();
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

        auto no_threads = std::thread::hardware_concurrency();
        size_t batch_size = std::round(total / no_threads);

        std::vector<std::thread> threads(no_threads);
        std::vector<std::tuple<float, float, float, float>> thread_results(no_threads);

        for (int t = 0; t < no_threads; t++) {
            threads[t] = std::thread([&, t]() {
                size_t start = t * batch_size;
                size_t end = std::min((t + 1) * batch_size, total);

                float local_loss = 0.0f;
                float local_accuracy = 0.0f;
                float local_true_positives = 0.0f;
                float local_false_positives = 0.0f;
                float local_false_negatives = 0.0f;

                for (size_t i = start; i < end; i++) {
                    auto example = test_data[i];
                    // get prediction and compute loss
                    Tensor3d prediction = predict(example.sequence, false);
                    local_loss += this->loss->compute(prediction, example.target);

                    // get predicted and actual class (assuming binary classification)
                    bool predicted_positive = prediction(0, 0, 0) > prediction(0, 1, 0);
                    bool actual_positive = example.target(0, 0, 0) > example.target(0, 1, 0);

                    // update metrics
                    if (predicted_positive == actual_positive) {
                        local_accuracy += 1.0f;
                    }

                    if (predicted_positive && actual_positive) {
                        local_true_positives += 1.0f;
                    } else if (predicted_positive && !actual_positive) {
                        local_false_positives += 1.0f;
                    } else if (!predicted_positive && actual_positive) {
                        local_false_negatives += 1.0f;
                    }
                }

                thread_results[t] = std::make_tuple(local_loss, local_accuracy, local_true_positives, local_false_positives);
            });
        }

        // wait for all threads to complete
        for (auto& thread : threads) {
            thread.join();
        }

        // aggregate results from all threads
        for (const auto& result : thread_results) {
            metrics.loss += std::get<0>(result);
            metrics.accuracy += std::get<1>(result);
            true_positives += std::get<2>(result);
            false_positives += std::get<3>(result);
        }

        // calculate final metrics
        metrics.loss /= total;
        metrics.accuracy /= total;

        // calculate precision, recall and f1 score
        metrics.precision = true_positives / (true_positives + false_positives);
        metrics.recall = true_positives / (true_positives + false_negatives);
        metrics.f1_score = 2 * (metrics.precision * metrics.recall) / (metrics.precision + metrics.recall);

        return metrics;
    }

    // fixme do not save attention layer
    void save_model(const std::string& filepath) const {
        std::ofstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("could not open file for saving: " + filepath);
        }

        // save model architecture parameters
        file.write(reinterpret_cast<const char*>(&input_size), sizeof(input_size));
        file.write(reinterpret_cast<const char*>(&hidden_size), sizeof(hidden_size));
        file.write(reinterpret_cast<const char*>(&attention_size), sizeof(attention_size));
        file.write(reinterpret_cast<const char*>(&output_size), sizeof(output_size));

        // save GRU parameters
        gru.W_z.save_to_file(file);
        gru.U_z.save_to_file(file);
        gru.b_z.save_to_file(file);
        gru.W_r.save_to_file(file);
        gru.U_r.save_to_file(file);
        gru.b_r.save_to_file(file);
        gru.W_h.save_to_file(file);
        gru.U_h.save_to_file(file);
        gru.b_h.save_to_file(file);

        // save attention parameters
        attention.weights.save_to_file(file);
        attention.scoring_vector.save_to_file(file);

        // save number of MLP layers
        size_t num_layers = mlp.layers.size();
        file.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));

        // save MLP parameters
        for (const auto& layer : mlp.layers) {
            // save layer dimensions and parameters
            layer.weights.save_to_file(file);
            layer.bias.save_to_file(file);

            // save activation function name
            size_t name_length = layer.activation_function.length();
            file.write(reinterpret_cast<const char*>(&name_length), sizeof(name_length));
            file.write(layer.activation_function.c_str(), name_length);
        }
    }

    static Predictor load_model(const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("could not open file for loading: " + filepath);
        }

        // load model architecture parameters
        size_t input_size, hidden_size, output_size, attention_size;
        file.read(reinterpret_cast<char*>(&input_size), sizeof(input_size));
        file.read(reinterpret_cast<char*>(&hidden_size), sizeof(hidden_size));
        file.read(reinterpret_cast<char*>(&attention_size), sizeof(attention_size));
        file.read(reinterpret_cast<char*>(&output_size), sizeof(output_size));

        // create predictor with loaded dimensions
        std::vector<int> mlp_topology = {static_cast<int>(hidden_size)};  // will be cleared and populated later just necessary to initialise MLP 
        Predictor predictor(input_size, hidden_size, attention_size, output_size, mlp_topology);  

        // load GRU parameters
        predictor.gru.W_z.load_from_file(file);
        predictor.gru.U_z.load_from_file(file);
        predictor.gru.b_z.load_from_file(file);
        predictor.gru.W_r.load_from_file(file);
        predictor.gru.U_r.load_from_file(file);
        predictor.gru.b_r.load_from_file(file);
        predictor.gru.W_h.load_from_file(file);
        predictor.gru.U_h.load_from_file(file);
        predictor.gru.b_h.load_from_file(file);

        // load attention parameters
        predictor.attention.weights.load_from_file(file);
        predictor.attention.scoring_vector.load_from_file(file);

        // load number of MLP layers
        size_t num_layers;
        file.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));

        // clear existing layers and load new ones
        predictor.mlp.layers.clear();

        // load MLP parameters
        for (size_t i = 0; i < num_layers; ++i) {
            Layer layer(1, 1);  // temporary dimensions, will be overwritten
            layer.weights.load_from_file(file);
            layer.bias.load_from_file(file);

            // load activation function name
            size_t name_length;
            file.read(reinterpret_cast<char*>(&name_length), sizeof(name_length));
            std::vector<char> name_buffer(name_length);
            file.read(name_buffer.data(), name_length);
            layer.activation_function = std::string(name_buffer.data(), name_length);

            predictor.mlp.layers.push_back(layer);
        }

        return predictor;
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
            Tensor3d target;
            if (sentiment == 1) {
                target = Tensor3d(1, 2, 1, std::vector<float>{1, 0});
            } else {
                target = Tensor3d(1, 2, 1, std::vector<float>{0, 1});
            }

            // tokenise review and create sequence tensor
            std::vector<Tensor3d> sequence = tokeniser.string_to_embeddings(review);

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

    const size_t batch_size = 100;
    const size_t epochs = 50;

    // load training and test data
    BatchDataLoader training_loader("../data/imdb_clean_train.csv", tokeniser, batch_size);
    BatchDataLoader test_loader("../data/imdb_clean_test.csv", tokeniser, 200);

    const int input_features = 100;
    const size_t hidden_size = 128;
    const size_t output_size = 2;
    const size_t attention_size = 128;
    const std::vector<int> mlp_topology = {static_cast<int>(hidden_size), 64, 32, static_cast<int>(output_size)};
    const std::vector<std::string> mlp_activation_functions = {"relu", "relu", "softmax"};

    Predictor predictor(input_features, hidden_size, attention_size, output_size, mlp_topology, mlp_activation_functions);
    predictor.set_gru_optimiser(std::make_unique<GRUAdamWOptimiser>(0.001, 0.9, 0.999, 1e-6, 0.003, 1.0));
    predictor.set_attention_optimiser(std::make_unique<AttentionAdamWOptimiser>(0.001, 0.9, 0.999, 1e-6, 0.003, 1.0));
    predictor.set_mlp_optimiser(std::make_unique<MLPAdamWOptimiser>(0.001, 0.9, 0.999, 1e-6, 0.003, 1.0));
    predictor.set_loss(std::make_unique<CrossEntropyLoss>());

    predictor.train_with_batches(training_loader, test_loader, epochs);

    return 0;
}