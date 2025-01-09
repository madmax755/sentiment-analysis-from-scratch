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

// sigmoid activation function
double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// sigmoid derivative
double sigmoid_derivative(double x) {
    double s = sigmoid(x);
    return s * (1.0 - s);
}

// relu activation function
double relu(double x) { return std::max(x, 0.0); }

// relu derivative
double relu_derivative(double x) { return (x > 0) ? 1.0 : 0.0; }

class Matrix {
   public:
    std::vector<std::vector<double>> data;  // 2D vector to store matrix data
    size_t rows, cols;                      // dimensions of the matrix

    /**
     * @brief Constructs a Matrix object with the specified dimensions.
     * @param rows Number of rows in the matrix.
     * @param cols Number of columns in the matrix.
     */
    Matrix(size_t rows, size_t cols) : rows(rows), cols(cols) {
        data.resize(rows,
                    std::vector<double>(cols, 0.0));  // resise the data vector to have 'rows' elements with a vector as the value
    }

    /**
     * @brief Initializes the matrix with random values between -1 and 1.
     */
    void uniform_initialise() {
        std::random_device rd;                            // obtain a random number from hardware
        std::mt19937 gen(rd());                           // seed the generator
        std::uniform_real_distribution<> dis(-1.0, 1.0);  // define the range
        for (auto& row : data) {
            for (auto& elem : row) {
                elem = dis(gen);  // generate random number
            }
        }
    }

    /**
     * @brief Initializes the matrix with zeros.
     */
    void zero_initialise() {
        for (auto& row : data) {
            for (auto& elem : row) {
                elem = 0;  // generate random number
            }
        }
    }

    /**
     * @brief Initializes the matrix using Xavier initialization method.
     * Suitable for sigmoid activation functions.
     */
    void xavier_initialize() {
        std::random_device rd;
        std::mt19937 gen(rd());
        double limit = sqrt(6.0 / (rows + cols));
        std::uniform_real_distribution<> dis(-limit, limit);
        for (auto& row : data) {
            for (auto& elem : row) {
                elem = dis(gen);
            }
        }
    }

    /**
     * @brief Initializes the matrix using He initialization method.
     * Suitable for ReLU activation functions.
     */
    void he_initialise() {
        std::random_device rd;
        std::mt19937 gen(rd());
        double std_dev = sqrt(2.0 / cols);
        std::normal_distribution<> dis(0, std_dev);
        for (auto& row : data) {
            for (auto& elem : row) {
                elem = dis(gen);
            }
        }
    }

    /**
     * @brief Overloads the multiplication operator for matrix multiplication.
     * @param other The matrix to multiply with.
     * @return The resulting matrix after multiplication.
     */
    Matrix operator*(const Matrix& other) const {
        if (cols != other.rows) {
            std::cerr << "Attempted to multiply matrices of incompatible dimensions: "
                      << "(" << rows << "x" << cols << ") * (" << other.rows << "x" << other.cols << ")" << std::endl;
            throw std::invalid_argument("Matrix dimensions don't match for multiplication");
        }
        Matrix result(rows, other.cols);
        // weird loop order (k before j) makes more cache friendly
        for (size_t i = 0; i < rows; i++) {
            for (size_t k = 0; k < cols; k++) {
                for (size_t j = 0; j < other.cols; j++) {
                    result.data[i][j] += data[i][k] * other.data[k][j];
                }
            }
        }
        return result;
    }

    /**
     * @brief Overloads the addition operator for element-wise matrix addition.
     * @param other The matrix to add.
     * @return The resulting matrix after addition.
     */
    Matrix operator+(const Matrix& other) const {
        if (rows != other.rows or cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions don't match for addition");
        }
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] + other.data[i][j];
            }
        }
        return result;
    }

    /**
     * @brief Overloads the subtraction operator for element-wise matrix subtraction.
     * @param other The matrix to subtract.
     * @return The resulting matrix after subtraction.
     */
    Matrix operator-(const Matrix& other) const {
        if (rows != other.rows or cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions don't match for subtraction");
        }
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] - other.data[i][j];
            }
        }
        return result;
    }

    /**
     * @brief Overloads the multiplication operator for scalar multiplication.
     * @param scalar The scalar value to multiply with.
     * @return The resulting matrix after scalar multiplication.
     */
    Matrix operator*(double scalar) const {
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] * scalar;
            }
        }
        return result;
    }

    /**
     * @brief Computes the transpose of the matrix.
     * @return The transposed matrix.
     */
    Matrix transpose() const {
        Matrix result(cols, rows);
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                result.data[j][i] = data[i][j];
            }
        }
        return result;
    }

    /**
     * @brief Computes the Hadamard product (element-wise multiplication) of two matrices.
     * @param other The matrix to perform Hadamard product with.
     * @return The resulting matrix after Hadamard product.
     */
    Matrix hadamard(const Matrix& other) const {
        if (rows != other.rows or cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions don't match for Hadamard product");
        }
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] * other.data[i][j];
            }
        }
        return result;
    }

    /**
     * @brief Applies a function to every element in the matrix.
     * @param func A function pointer to apply to each element.
     * @return The resulting matrix after applying the function.
     */
    Matrix apply(double (*func)(double)) const {
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                result.data[i][j] = func(data[i][j]);
            }
        }
        return result;
    }

    /**
     * @brief Applies a function to every element in the matrix.
     * @tparam Func The type of the callable object.
     * @param func A callable object to apply to each element.
     * @return The resulting matrix after applying the function.
     */
    template <typename Func>
    Matrix apply(Func func) const {
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                result.data[i][j] = func(data[i][j]);
            }
        }
        return result;
    }

    /**
     * @brief Applies the softmax function to the matrix.
     * @return The resulting matrix after applying softmax.
     */
    Matrix softmax() const {
        Matrix result(rows, cols);

        for (size_t j = 0; j < cols; ++j) {
            double max_val = -std::numeric_limits<double>::infinity();
            for (size_t i = 0; i < rows; ++i) {
                max_val = std::max(max_val, data[i][j]);
            }

            double sum = 0.0;
            for (size_t i = 0; i < rows; ++i) {
                result.data[i][j] = std::exp(data[i][j] - max_val);
                sum += result.data[i][j];
            }

            for (size_t i = 0; i < rows; ++i) {
                result.data[i][j] /= sum;
            }
        }

        return result;
    }

    friend std::ostream& operator<<(std::ostream& os, const Matrix& matrix) {
        os << "[";
        for (size_t i = 0; i < matrix.rows; ++i) {
            if (i > 0) os << " ";
            os << "[";
            for (size_t j = 0; j < matrix.cols; ++j) {
                os << matrix.data[i][j];
                if (j < matrix.cols - 1) os << ", ";
            }
            os << "]";
            if (i < matrix.rows - 1) os << "\n";
        }
        os << "]";
        return os;
    }
};

struct TrainingExample {
    std::vector<Matrix> sequence;
    Matrix target;

    TrainingExample() : target(1, 1) {}  // initialize target with size 1x1 as matrix does not have default constructor
};

// gradient storage struct for GRUCell -- defined globally to avoid circular dependency
struct GRUGradients {
    Matrix dW_z, dU_z, db_z;  // update gate gradients
    Matrix dW_r, dU_r, db_r;  // reset gate gradients
    Matrix dW_h, dU_h, db_h;  // hidden state gradients
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
    GRUGradients operator*(double scalar) const {
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
        Matrix z, r, h_candidate, h;
        Matrix h_prev;
        Matrix x;

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

    // store the gradients and the gradient of the hidden state from the previous timestep
    struct BackwardResult {
        GRUGradients grads;
        Matrix dh_prev;

        BackwardResult(GRUGradients g, Matrix h) : grads(g), dh_prev(h) {}
    };

    // compute gradients for a single timestep
    BackwardResult backward(const Matrix& delta_h_t, size_t t) {
        if (t >= time_steps.size()) {
            throw std::runtime_error("Time step index out of bounds");
        }

        // get the stored states for this timestep
        const TimeStep& step = time_steps[t];

        // initialize gradients for this timestep
        GRUGradients timestep_grads(input_size, hidden_size);

        // 1. Hidden state gradients
        Matrix one_matrix(delta_h_t.rows, delta_h_t.cols);
        for (size_t i = 0; i < one_matrix.rows; i++)
            for (size_t j = 0; j < one_matrix.cols; j++) one_matrix.data[i][j] = 1.0;

        Matrix dh_tilde = delta_h_t.hadamard(one_matrix - step.z);
        Matrix dz = delta_h_t.hadamard(step.h_prev - step.h_candidate);

        // 2. Candidate state gradients
        Matrix dg = dh_tilde.hadamard(step.h_candidate.apply([](double x) { return 1.0 - x * x; })  // tanh derivative
        );

        timestep_grads.dW_h = dg * step.x.transpose();
        timestep_grads.dU_h = dg * (step.r.hadamard(step.h_prev)).transpose();
        timestep_grads.db_h = dg;

        Matrix dx_t = timestep_grads.dW_h.transpose() * dg;
        Matrix dr = (timestep_grads.dU_h.transpose() * dg).hadamard(step.h_prev);
        Matrix dh_prev = (timestep_grads.dU_h.transpose() * dg).hadamard(step.r);

        // 3. Reset gate gradients
        Matrix dr_total = dr.hadamard(step.r.apply(sigmoid_derivative));

        timestep_grads.dW_r = dr_total * step.x.transpose();
        timestep_grads.dU_r = dr_total * step.h_prev.transpose();
        timestep_grads.db_r = dr_total;

        dx_t = dx_t + timestep_grads.dW_r.transpose() * dr_total;
        dh_prev = dh_prev + timestep_grads.dU_r.transpose() * dr_total;

        // 4. Update gate gradients
        Matrix dz_total = dz.hadamard(step.z.apply(sigmoid_derivative));

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
    Matrix W_z;  // update gate weights for input
    Matrix U_z;  // update gate weights for hidden state
    Matrix b_z;  // update gate bias

    Matrix W_r;  // reset gate weights for input
    Matrix U_r;  // reset gate weights for hidden state
    Matrix b_r;  // reset gate bias

    Matrix W_h;  // candidate hidden state weights for input
    Matrix U_h;  // candidate hidden state weights for hidden state
    Matrix b_h;  // candidate hidden state bias

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
        // initialize weights using Xavier initialization
        W_z.xavier_initialize();
        U_z.xavier_initialize();
        W_r.xavier_initialize();
        U_r.xavier_initialize();
        W_h.xavier_initialize();
        U_h.xavier_initialize();

        // biases are initialised to zero by default
    }

    // get final hidden state
    Matrix get_last_hidden_state() const {
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
    Matrix forward(const Matrix& x, const Matrix& h_prev) {
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
        step.h = step.z.hadamard(h_prev) + (step.z.apply([](double x) { return 1.0 - x; }).hadamard(step.h_candidate));

        time_steps.push_back(step);
        return step.h;
    }

    GRUGradients backpropagate(const Matrix& final_gradient) {
        Matrix dh_next = final_gradient;
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
    Matrix weights;
    Matrix bias;
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
            weights.xavier_initialize();
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
    Matrix feedforward(const Matrix& input) {
        Matrix z = weights * input + bias;
        Matrix output(z.rows, z.cols);
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
    std::vector<Matrix> feedforward_backprop(const Matrix& input) const {
        Matrix z = weights * input + bias;
        Matrix output(z.rows, z.cols);
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
    virtual double compute(const Matrix& predicted, const Matrix& target) const = 0;

    // Compute the derivative of the loss with respect to the predicted values
    virtual Matrix derivative(const Matrix& predicted, const Matrix& target) const = 0;
};

class CrossEntropyLoss : public Loss {
   public:
    double compute(const Matrix& predicted, const Matrix& target) const override {
        double loss = 0.0;
        for (size_t i = 0; i < predicted.rows; ++i) {
            for (size_t j = 0; j < predicted.cols; ++j) {
                // Add small epsilon to avoid log(0)
                loss -= target.data[i][j] * std::log(predicted.data[i][j] + 1e-10);
            }
        }
        return loss / predicted.cols;  // Average over batch
    }

    Matrix derivative(const Matrix& predicted, const Matrix& target) const override {
        // For cross entropy with softmax, the derivative simplifies to (predicted - target)
        return predicted - target;
    }
};

class MSELoss : public Loss {
   public:
    double compute(const Matrix& predicted, const Matrix& target) const override {
        double loss = 0.0;
        for (size_t i = 0; i < predicted.rows; ++i) {
            for (size_t j = 0; j < predicted.cols; ++j) {
                double diff = predicted.data[i][j] - target.data[i][j];
                loss += diff * diff;
            }
        }
        return loss / (2.0 * predicted.cols);  // Average over batch and divide by 2
    }

    Matrix derivative(const Matrix& predicted, const Matrix& target) const override {
        return (predicted - target) * (1.0 / predicted.cols);
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
    virtual void compute_and_apply_updates(std::vector<Layer>& layers, const std::vector<std::vector<Matrix>>& gradients) = 0;

    /**
     * @brief Virtual destructor for the MLPOptimiser class.
     */
    virtual ~MLPOptimiser() = default;

    struct GradientResult {
        std::vector<std::vector<Matrix>> gradients;  // list of layers, each layer has a list of weight and bias gradient matrices
        Matrix input_layer_gradient;  // gradient of the input layer - for more general use as parts of bigger architectures
        Matrix output;                // output of the network
    };

    /**
     * @brief Calculates gradients for a single example.
     * @param layers The layers of the neural network.
     * @param input The input matrix.
     * @param target The target matrix.
     * @return A GradientResult struct containing gradients and the output of the network.
     */
    virtual GradientResult calculate_gradient(const std::vector<Layer>& layers, const Matrix& input, const Matrix& target,
                                              const Loss& loss) {
        // forward pass
        std::vector<Matrix> activations = {input};
        std::vector<Matrix> preactivations = {input};

        for (const auto& layer : layers) {
            auto results = layer.feedforward_backprop(activations.back());
            activations.push_back(results[0]);
            preactivations.push_back(results[1]);
        }

        // backward pass
        int num_layers = layers.size();
        std::vector<Matrix> deltas;
        deltas.reserve(num_layers);

        // output layer error (δ^L = ∇_a C ⊙ σ'(z^L))
        Matrix output_delta = loss.derivative(activations.back(), target);
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
            Matrix delta = (layers[l + 1].weights.transpose() * deltas.back());
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
        std::vector<std::vector<Matrix>> gradients;
        for (int l = 0; l < num_layers; ++l) {
            Matrix weight_gradient = deltas[l] * activations[l].transpose();
            gradients.push_back({weight_gradient, deltas[l]});
        }

        // as we don't treat the input as a layer, we need to return the input layer errors separately
        Matrix input_delta = layers[0].weights.transpose() * deltas[0];

        // return a GradientResult struct for purposes of tracking loss
        return {gradients, input_delta, activations.back()};
    }

    /**
     * @brief Averages gradients from multiple examples.
     * @param batch_gradients A vector of gradients from multiple examples.
     * @return The averaged gradients.
     */
    std::vector<std::vector<Matrix>> average_gradients(const std::vector<std::vector<std::vector<Matrix>>>& batch_gradients) {
        std::vector<std::vector<Matrix>> avg_gradients;
        size_t num_layers = batch_gradients[0].size();
        size_t batch_size = batch_gradients.size();

        for (size_t l = 0; l < num_layers; ++l) {
            Matrix avg_weight_grad(batch_gradients[0][l][0].rows, batch_gradients[0][l][0].cols);
            Matrix avg_bias_grad(batch_gradients[0][l][1].rows, batch_gradients[0][l][1].cols);

            for (const auto& example_gradients : batch_gradients) {
                avg_weight_grad = avg_weight_grad + example_gradients[l][0];
                avg_bias_grad = avg_bias_grad + example_gradients[l][1];
            }

            avg_weight_grad = avg_weight_grad * (1.0 / batch_size);
            avg_bias_grad = avg_bias_grad * (1.0 / batch_size);

            avg_gradients.push_back({avg_weight_grad, avg_bias_grad});
        }

        return avg_gradients;
    }
};

class MLPSGDOptimiser : public MLPOptimiser {
   private:
    double learning_rate;
    std::vector<std::vector<Matrix>> velocity;

   public:
    /**
     * @brief Constructs an MLPSGDOptimiser object with the specified learning rate.
     * @param lr The learning rate (default: 0.1).
     */
    MLPSGDOptimiser(double lr = 0.1) : learning_rate(lr) {}

    /**
     * @brief Initializes the velocity vectors for SGD optimization.
     * @param layers The layers of the neural network.
     */
    void initialize_velocity(const std::vector<Layer>& layers) {
        velocity.clear();
        for (const auto& layer : layers) {
            velocity.push_back({Matrix(layer.weights.rows, layer.weights.cols), Matrix(layer.bias.rows, layer.bias.cols)});
        }
    }

    /**
     * @brief Computes and applies updates using Stochastic Gradient Descent.
     * @param layers The layers of the neural network to update.
     * @param gradients The gradients used for updating the layers.
     */
    void compute_and_apply_updates(std::vector<Layer>& layers, const std::vector<std::vector<Matrix>>& gradients) override {
        if (velocity.empty()) {
            initialize_velocity(layers);
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
    double learning_rate;
    double momentum;
    std::vector<std::vector<Matrix>> velocity;

   public:
    /**
     * @brief Constructs an MLPSGDMomentumOptimiser object with the specified learning rate and momentum.
     * @param lr The learning rate (default: 0.1).
     * @param mom The momentum coefficient (default: 0.9).
     */
    MLPSGDMomentumOptimiser(double lr = 0.1, double mom = 0.9) : learning_rate(lr), momentum(mom) {}

    /**
     * @brief Initializes the velocity vectors for SGD with Momentum optimization.
     * @param layers The layers of the neural network.
     */
    void initialize_velocity(const std::vector<Layer>& layers) {
        velocity.clear();
        for (const auto& layer : layers) {
            velocity.push_back({Matrix(layer.weights.rows, layer.weights.cols), Matrix(layer.bias.rows, layer.bias.cols)});
        }
    }

    /**
     * @brief Computes and applies updates using Stochastic Gradient Descent with Momentum.
     * @param layers The layers of the neural network to update.
     * @param gradients The gradients used for updating the layers.
     */
    void compute_and_apply_updates(std::vector<Layer>& layers, const std::vector<std::vector<Matrix>>& gradients) override {
        if (velocity.empty()) {
            initialize_velocity(layers);
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
    double learning_rate;
    double beta1;
    double beta2;
    double epsilon;
    int t;                               // timestep
    std::vector<std::vector<Matrix>> m;  // first moment
    std::vector<std::vector<Matrix>> v;  // second moment

   public:
    /**
     * @brief Constructs an MLPAdamOptimiser object with the specified parameters.
     * @param lr The learning rate (default: 0.001).
     * @param b1 The beta1 parameter (default: 0.9).
     * @param b2 The beta2 parameter (default: 0.999).
     * @param eps The epsilon parameter for numerical stability (default: 1e-8).
     */
    MLPAdamOptimiser(double lr = 0.001, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8)
        : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), t(0) {}

    /**
     * @brief Initializes the first and second moment vectors for Adam optimization.
     * @param layers The layers of the neural network.
     */
    void initialize_moments(const std::vector<Layer>& layers) {
        m.clear();
        v.clear();
        m.reserve(layers.size());
        v.reserve(layers.size());
        for (const auto& layer : layers) {
            m.push_back({Matrix(layer.weights.rows, layer.weights.cols), Matrix(layer.bias.rows, layer.bias.cols)});
            v.push_back({Matrix(layer.weights.rows, layer.weights.cols), Matrix(layer.bias.rows, layer.bias.cols)});
        }
    }

    /**
     * @brief Computes and applies updates using the Adam optimization algorithm.
     * @param layers The layers of the neural network to update.
     * @param gradients The gradients used for updating the layers.
     */
    void compute_and_apply_updates(std::vector<Layer>& layers, const std::vector<std::vector<Matrix>>& gradients) override {
        if (m.empty() or v.empty()) {
            initialize_moments(layers);
        }

        t++;  // increment timestep

        for (size_t l = 0; l < layers.size(); ++l) {
            for (int i = 0; i < 2; ++i) {  // 0 for weights, 1 for biases
                // update biased first moment estimate
                m[l][i] = m[l][i] * beta1 + gradients[l][i] * (1.0 - beta1);

                // update biased second raw moment estimate
                v[l][i] = v[l][i] * beta2 + gradients[l][i].hadamard(gradients[l][i]) * (1.0 - beta2);

                // compute bias-corrected first moment estimate
                Matrix m_hat = m[l][i] * (1.0 / (1.0 - std::pow(beta1, t)));

                // compute bias-corrected second raw moment estimate
                Matrix v_hat = v[l][i] * (1.0 / (1.0 - std::pow(beta2, t)));

                // compute the update
                Matrix update = m_hat.hadamard(v_hat.apply([this](double x) { return 1.0 / (std::sqrt(x) + epsilon); }));

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
    double learning_rate;
    double beta1;
    double beta2;
    double epsilon;
    double weight_decay;
    int t;                               // timestep
    std::vector<std::vector<Matrix>> m;  // first moment
    std::vector<std::vector<Matrix>> v;  // second moment

   public:
    /**
     * @brief Constructs an MLPAdamWOptimiser object with the specified parameters.
     * @param lr The learning rate (default: 0.001).
     * @param b1 The beta1 parameter (default: 0.9).
     * @param b2 The beta2 parameter (default: 0.999).
     * @param eps The epsilon parameter for numerical stability (default: 1e-8).
     * @param wd The weight decay parameter (default: 0.01).
     */
    MLPAdamWOptimiser(double lr = 0.001, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8, double wd = 0.01)
        : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), weight_decay(wd), t(0) {}

    /**
     * @brief Initializes the first and second moment vectors for AdamW optimization.
     * @param layers The layers of the neural network.
     */
    void initialize_moments(const std::vector<Layer>& layers) {
        m.clear();
        v.clear();
        m.reserve(layers.size());
        v.reserve(layers.size());
        for (const auto& layer : layers) {
            m.push_back({Matrix(layer.weights.rows, layer.weights.cols), Matrix(layer.bias.rows, layer.bias.cols)});
            v.push_back({Matrix(layer.weights.rows, layer.weights.cols), Matrix(layer.bias.rows, layer.bias.cols)});
        }
    }

    /**
     * @brief Computes and applies updates using the AdamW optimization algorithm.
     * @param layers The layers of the neural network to update.
     * @param gradients The gradients used for updating the layers.
     */
    void compute_and_apply_updates(std::vector<Layer>& layers, const std::vector<std::vector<Matrix>>& gradients) override {
        if (m.empty() || v.empty()) {
            initialize_moments(layers);
        }

        t++;  // increment timestep

        for (size_t l = 0; l < layers.size(); ++l) {
            for (int i = 0; i < 2; ++i) {  // 0 for weights, 1 for biases
                // update biased first moment estimate
                m[l][i] = m[l][i] * beta1 + gradients[l][i] * (1.0 - beta1);

                // update biased second raw moment estimate
                v[l][i] = v[l][i] * beta2 + gradients[l][i].hadamard(gradients[l][i]) * (1.0 - beta2);

                // compute bias-corrected first moment estimate
                Matrix m_hat = m[l][i] * (1.0 / (1.0 - std::pow(beta1, t)));

                // compute bias-corrected second raw moment estimate
                Matrix v_hat = v[l][i] * (1.0 / (1.0 - std::pow(beta2, t)));

                // compute the Adam update
                Matrix update = m_hat.hadamard(v_hat.apply([this](double x) { return 1.0 / (std::sqrt(x) + epsilon); }));

                // apply the update
                if (i == 0) {  // for weights
                    // apply weight decay
                    layers[l].weights = layers[l].weights * (1.0 - learning_rate * weight_decay);
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
    double learning_rate;

   public:
    GRUSGDOptimiser(double lr = 0.1) : learning_rate(lr) {}

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
    double learning_rate;
    double momentum;
    GRUGradients velocity;

   public:
    GRUSGDMomentumOptimiser(double lr = 0.1, double mom = 0.9)
        : learning_rate(lr), momentum(mom), velocity(0, 0) {}  // sizes will be set on first use

    void compute_and_apply_updates(GRUCell& gru, const GRUGradients& grads) override {
        // initialize velocity if needed
        if (velocity.dW_z.rows == 0) {
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
    double learning_rate;
    double beta1;
    double beta2;
    double epsilon;
    int t;
    GRUGradients m;
    GRUGradients v;

    void update_parameter(Matrix& param, Matrix& m_param, Matrix& v_param, const Matrix& grad) {
        // Update biased first moment estimate
        m_param = m_param * beta1 + grad * (1.0 - beta1);

        // Update biased second raw moment estimate
        v_param = v_param * beta2 + grad.hadamard(grad) * (1.0 - beta2);

        // Compute bias-corrected first moment estimate
        Matrix m_hat = m_param * (1.0 / (1.0 - std::pow(beta1, t)));

        // Compute bias-corrected second raw moment estimate
        Matrix v_hat = v_param * (1.0 / (1.0 - std::pow(beta2, t)));

        // Update parameters
        Matrix denom = v_hat.apply([this](double x) { return 1.0 / (std::sqrt(x) + epsilon); });
        Matrix update = m_hat.hadamard(denom);
        param = param - update * learning_rate;
    }

   public:
    GRUAdamOptimiser(double lr = 0.001, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8)
        : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), t(0), m(0, 0), v(0, 0) {}

    void compute_and_apply_updates(GRUCell& gru, const GRUGradients& grads) override {
        if (m.dW_z.rows == 0) {
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
    double learning_rate;
    double beta1;
    double beta2;
    double epsilon;
    double weight_decay;
    int t;
    GRUGradients m;
    GRUGradients v;

    void update_parameter(Matrix& param, Matrix& m_param, Matrix& v_param, const Matrix& grad, bool apply_weight_decay = true) {
        // Weight decay should be applied to the parameter directly
        if (apply_weight_decay) {
            param = param * (1.0 - learning_rate * weight_decay);
        }

        // Update biased first moment estimate
        m_param = m_param * beta1 + grad * (1.0 - beta1);

        // Update biased second raw moment estimate
        v_param = v_param * beta2 + grad.hadamard(grad) * (1.0 - beta2);

        // Compute bias-corrected first moment estimate
        Matrix m_hat = m_param * (1.0 / (1.0 - std::pow(beta1, t)));

        // Compute bias-corrected second raw moment estimate
        Matrix v_hat = v_param * (1.0 / (1.0 - std::pow(beta2, t)));

        // Update parameters
        Matrix denom = v_hat.apply([this](double x) { return 1.0 / (std::sqrt(x) + epsilon); });
        Matrix update = m_hat.hadamard(denom);
        param = param - update * learning_rate;
    }

   public:
    GRUAdamWOptimiser(double lr = 0.001, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8, double wd = 0.01)
        : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), weight_decay(wd), t(0), m(0, 0), v(0, 0) {}

    void compute_and_apply_updates(GRUCell& gru, const GRUGradients& grads) override {
        if (m.dW_z.rows == 0) {
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
    Matrix feedforward(const Matrix& input) {
        Matrix current = input;
        for (auto& layer : layers) {
            current = layer.feedforward(current);
        }
        return current;
    }

    size_t get_index_of_max_element_in_nx1_matrix(const Matrix& matrix) const {
        size_t index = 0;
        double max_value = matrix.data[0][0];
        for (size_t i = 1; i < matrix.rows; ++i) {
            if (matrix.data[i][0] > max_value) {
                index = i;
                max_value = matrix.data[i][0];
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
    Predictor(size_t input_size, size_t hidden_size, size_t output_size, std::vector<int> mlp_topology, std::vector<std::string> mlp_activation_functions = {})
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
    void update_parameters(const GRUGradients& gru_grads, const std::vector<std::vector<Matrix>>& mlp_grads) {
        if (!gru_optimiser) {
            throw std::runtime_error("no optimiser set");
        }
        gru_optimiser->compute_and_apply_updates(gru, gru_grads);
        mlp_optimiser->compute_and_apply_updates(mlp.layers, mlp_grads);
    }

    // process sequence and return prediction (full feedforward pass)
    Matrix predict(const std::vector<Matrix>& input_sequence) {
        Matrix h_t(hidden_size, 1);

        // process sequence through GRU
        for (const auto& x : input_sequence) {
            h_t = gru.forward(x, h_t);
        }

        // final linear layer
        return mlp.feedforward(h_t);
    }

    // process sequence and return final hidden state
    Matrix feedforward_gru(const std::vector<Matrix>& input_sequence) {
        Matrix h_t(hidden_size, 1);

        // process sequence through GRU
        for (const auto& x : input_sequence) {
            h_t = gru.forward(x, h_t);
        }

        // final hidden state
        return h_t;
    }

    // gets the gradients for a single training example
    std::pair<GRUGradients, std::vector<std::vector<Matrix>>> compute_gradients(const std::vector<Matrix>& input_sequence,
                                                                                const Matrix& target) {
        // forward pass
        Matrix final_hidden_state = feedforward_gru(input_sequence);

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
                std::vector<std::vector<std::vector<Matrix>>> accumulated_mlp_gradients;
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
                averaged_gru_gradients = averaged_gru_gradients * (1.0 / (batch_end - batch_start));

                // average accumulated mlp gradients
                std::vector<std::vector<Matrix>> averaged_mlp_gradients =
                    mlp_optimiser->average_gradients(accumulated_mlp_gradients);

                // update GRU parameters using optimiser once batch gradients are averaged
                update_parameters(averaged_gru_gradients, averaged_mlp_gradients);

                std::cout << "\rBatch " << i / batch_size << "/" << no_examples / batch_size << " complete" << std::flush;
            }
            std::cout << "\rEpoch " << epoch << "/" << epochs << " complete" << std::endl;
            auto test_metrics = evaluate(test_data);
            std::cout << test_metrics << std::endl;
        }
    }

    // add this to your Predictor class
    struct EvaluationMetrics {
        double mse;
        double mae;
        double rmse;
        double profit_loss;
        double accuracy;
        int total_trades;
        double avg_trade_return;  // added: average return per trade

        friend std::ostream& operator<<(std::ostream& os, const EvaluationMetrics& metrics) {
            os << "----------------\n"
               << "MSE: " << metrics.mse << "\n"
               << "MAE: " << metrics.mae << "\n"
               << "RMSE: " << metrics.rmse << "\n"
               << "Profit/Loss: " << (metrics.profit_loss * 100) << "%\n"
               << "Direction Accuracy: " << (metrics.accuracy * 100) << "%\n"
               << "Total Trades: " << metrics.total_trades << "\n"
               << "Avg Trade Return: " << (metrics.avg_trade_return * 100) << "%\n"
               << "----------------";
            return os;
        }
    };

    EvaluationMetrics evaluate(const std::vector<TrainingExample>& test_data) {
        double total_squared_error = 0.0;
        double total_absolute_error = 0.0;
        size_t total_examples = test_data.size();

        // trading metrics
        double portfolio_value = 1.0;
        double peak_value = 1.0;
        double max_drawdown = 0.0;
        int correct_predictions = 0;
        int total_trades = 0;

        std::vector<double> trade_returns;
        const double max_position_size = 0.2;  // maximum 20% of portfolio per trade

        for (const auto& example : test_data) {
            Matrix prediction = predict(example.sequence);
            double predicted_return = prediction.data[0][0];
            double actual_return = example.target.data[0][0];

            // compute errors for traditional metrics
            double error = predicted_return - actual_return;
            total_squared_error += error * error;
            total_absolute_error += std::abs(error);

            // trading simulation with proportional betting
            if (std::abs(predicted_return) > 0.01) {  // minimum threshold for trading
                total_trades++;

                // calculate position size based on prediction confidence
                double confidence = std::abs(predicted_return);
                double position_size = std::min(confidence, max_position_size);

                // check prediction direction
                if ((predicted_return > 0 && actual_return > 0) || (predicted_return < 0 && actual_return < 0)) {
                    correct_predictions++;
                }

                // simulate trade with position sizing
                double trade_return;
                if (predicted_return > 0) {
                    // long position
                    trade_return = position_size * actual_return;
                } else {
                    // short position
                    trade_return = position_size * (-actual_return);
                }

                // update portfolio value
                portfolio_value *= (1.0 + trade_return);
                trade_returns.push_back(trade_return);
            }
        }

        // compute basic metrics
        double mse = total_squared_error / total_examples;
        double mae = total_absolute_error / total_examples;
        double rmse = std::sqrt(mse);
        double accuracy = total_trades > 0 ? static_cast<double>(correct_predictions) / total_trades : 0.0;
        double profit_loss = portfolio_value - 1.0;

        // compute average trade return
        double avg_trade_return = 0.0;
        if (!trade_returns.empty()) {
            avg_trade_return = std::accumulate(trade_returns.begin(), trade_returns.end(), 0.0) / trade_returns.size();
        }

        return {mse, mae, rmse, profit_loss, accuracy, total_trades, avg_trade_return};
    }
};

std::vector<TrainingExample> generate_sine_training_data(int num_samples, int sequence_length, double sampling_frequency = 0.1) {
    std::vector<TrainingExample> training_data;

    // generate a longer sine wave
    std::vector<double> sine_wave;
    for (int i = 0; i < num_samples + sequence_length; i++) {
        double x = i * sampling_frequency;
        sine_wave.push_back(std::sin(x));
    }

    // create sliding window examples
    for (int i = 0; i < num_samples; i++) {
        TrainingExample example;

        // create input sequence
        for (int j = 0; j < sequence_length; j++) {
            Matrix input(1, 1);  // single feature (sine value)
            input.data[0][0] = sine_wave[i + j];
            example.sequence.push_back(input);
        }

        // create target (next value in sequence)
        Matrix target(1, 1);
        target.data[0][0] = sine_wave[i + sequence_length];
        example.target = target;

        training_data.push_back(example);
    }

    return training_data;
}

std::pair<std::vector<TrainingExample>, size_t> load_stock_data(const std::string& filename, size_t sequence_length = 20) {
    std::vector<TrainingExample> training_data;
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file: " + filename);
    }

    // read header to determine number of features
    std::getline(file, line);
    std::stringstream header_ss(line);
    std::string header_value;
    size_t n_features = 0;
    while (std::getline(header_ss, header_value, ',')) {
        if (!header_value.empty()) {
            n_features++;
        }
    }

    if (n_features == 0) {
        throw std::runtime_error("no features found in header");
    }

    // read all lines into temporary storage
    std::vector<std::vector<double>> all_data;
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string value;
        std::vector<double> features;

        while (std::getline(ss, value, ',')) {
            try {
                if (!value.empty()) {
                    features.push_back(std::stod(value));
                }
            } catch (const std::invalid_argument& e) {
                std::cerr << "warning: invalid number format in line: " << line << std::endl;
                continue;
            }
        }

        if (features.size() == n_features) {
            all_data.push_back(features);
        } else {
            std::cerr << "warning: incorrect number of features (" << features.size() << "/" << n_features
                      << ") in line: " << line << std::endl;
        }
    }

    // create sequences
    for (size_t i = 0; i + sequence_length < all_data.size(); i++) {
        TrainingExample example;

        // create sequence using all features
        for (size_t j = 0; j < sequence_length; j++) {
            Matrix input(n_features, 1);
            for (size_t f = 0; f < n_features; f++) {
                input.data[f][0] = all_data[i + j][f];
            }
            example.sequence.push_back(input);
        }

        // use next Return (first column) as target
        example.target = Matrix(1, 1);
        example.target.data[0][0] = all_data[i + sequence_length][0];

        training_data.push_back(example);
    }

    if (training_data.empty()) {
        throw std::runtime_error("no valid training examples could be created from file");
    }

    return {training_data, n_features};
}

int main() {
    // generate training data
    auto [stock_data, n_features] = load_stock_data("stock_data/AAPL_data_normalised.csv", 100);
    std::shuffle(stock_data.begin(), stock_data.end(), std::mt19937(std::random_device()()));

    // split data into training and test sets
    size_t split_point = static_cast<size_t>(stock_data.size() * 0.8);
    auto training_data = std::vector<TrainingExample>(stock_data.begin(), stock_data.begin() + split_point);
    auto test_data = std::vector<TrainingExample>(stock_data.begin() + split_point, stock_data.end());

    size_t input_features = n_features;
    size_t hidden_size = 128;
    size_t output_size = 1;
    std::vector<int> mlp_topology = {static_cast<int>(hidden_size), 32, static_cast<int>(output_size)};
    std::vector<std::string> mlp_activation_functions = {"sigmoid", "sigmoid", "none"};


    Predictor predictor(input_features, hidden_size, output_size, mlp_topology);
    predictor.set_gru_optimiser(std::make_unique<GRUAdamWOptimiser>());
    predictor.set_mlp_optimiser(std::make_unique<MLPAdamWOptimiser>());
    predictor.set_loss(std::make_unique<MSELoss>());

    predictor.train(training_data, test_data, 75, 50);

    // print some example predictions
    std::cout << "\nSample predictions:" << std::endl;
    for (size_t i = 0; i < std::min(size_t(10), test_data.size()); i++) {
        Matrix prediction = predictor.predict(test_data[i].sequence);
        std::cout << "Predicted: " << prediction.data[0][0] << " Actual: " << test_data[i].target.data[0][0] << std::endl;
    }
}