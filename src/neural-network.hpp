#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK

#include "matrix.hpp"
#include <vector>

/**
 * @class NeuralNetwork
 * @brief A simple neural network implementation
 * 
 * This class implements a basic feedforward neural network with:
 * - Customizable layer architecture
 * - Sigmoid activation function for hidden layers
 * - Backpropagation for training with gradient descent
 * - Mean squared error for loss calculation
 */
class NeuralNetwork {
private:
    std::vector<Matrix> weights;    // Weight matrices for each layer connection
    std::vector<Matrix> biases;     // Bias vectors for each layer
    std::vector<size_t> layers;     // Number of neurons in each layer
    double learning_rate;           // Learning rate for gradient descent

public:
    /**
     * @brief Constructor for NeuralNetwork
     * @param layer_sizes Vector containing the number of neurons in each layer
     * @param lr Learning rate for training (default 0.01)
     */
    NeuralNetwork(std::vector<size_t> layer_sizes, double lr = 0.01) {
        // Both of these can be set through the initializer list for future speedup
        learning_rate = lr;
        layers = layer_sizes;
        
        // Initialize weights and biases for each layer connection
        for (size_t i = 0; i < layers.size() - 1; i++) {
            // Weight matrix: (next_layer_size) x (current_layer_size)
            Matrix w(layers[i + 1], layers[i]);
            w.randomize(-1.0, 1.0);  // Initialize with random values between -1 and 1
            weights.push_back(w);
            
            // Bias matrix: (next_layer_size) x 1
            Matrix b(layers[i + 1], 1);
            b.randomize(-1.0, 1.0);  // Initialize with random values between -1 and 1
            biases.push_back(b);
        }
    }

    /**
     * @brief Forward propagation through the network
     * @param input Input data matrix (must match input layer size)
     * @return Output matrix from the output layer
     */
    Matrix forward(const Matrix& input) {
        Matrix activation = input;
        
        // Process each layer
        for (size_t i = 0; i < weights.size(); i++) {
            // Linear transformation: z = Wx + b
            activation = weights[i] * activation + biases[i];
            
            // Apply sigmoid activation (except for output layer)
            if (i < weights.size() - 1) { 
                activation.sigmoid();
            }
        }
        
        return activation;
    }

    /**
     * @brief Train the network using backpropagation
     * @param input Input data matrix
     * @param target Expected output matrix
     */
    void train(const Matrix& input, const Matrix& target) {
        // Forward pass - collect activations and z-values
        std::vector<Matrix> activations;  // Store activations of each layer
        std::vector<Matrix> z_values;     // Store pre-activation values
        
        Matrix activation = input;
        activations.push_back(activation);
        
        // Forward propagation
        for (size_t i = 0; i < weights.size(); i++) {
            Matrix z = weights[i] * activation + biases[i];
            z_values.push_back(z);
            
            if (i < weights.size() - 1) { // Apply sigmoid to hidden layers
                activation = z;
                activation.sigmoid();
            } else {
                activation = z;  // No activation for output layer (linear output)
            }
            activations.push_back(activation);
        }
        
        // Backward pass - calculate errors and update weights/biases
        Matrix error = activations.back() - target;  // Output layer error
        
        // Update output layer
        size_t output_layer = weights.size() - 1;
        weights[output_layer] = weights[output_layer] - (error * activations[output_layer].transpose()) * learning_rate;
        biases[output_layer] = biases[output_layer] - error * learning_rate;
        
        // Update hidden layers
        for (int i = output_layer - 1; i >= 0; i--) {
            // Propagate error backwards
            error = weights[i + 1].transpose() * error;
            
            // Calculate sigmoid derivative: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
            Matrix sigmoid_derivative = activations[i + 1];
            sigmoid_derivative.square(); // sigmoid²(x)
            Matrix ones(sigmoid_derivative.getRows(), sigmoid_derivative.getCols(), 1.0);
            Matrix derivative = ones - sigmoid_derivative;
            
            // Apply derivative to error
            error = error;
            for (size_t j = 0; j < error.getRows(); j++) {
                for (size_t k = 0; k < error.getCols(); k++) {
                    error(j, k) *= derivative(j, k);
                }
            }
            
            // Update weights and biases
            weights[i] = weights[i] - (error * activations[i].transpose()) * learning_rate;
            biases[i] = biases[i] - error * learning_rate;
        }
    }

    /**
     * @brief Calculate mean squared error
     * @param predicted Predicted output
     * @param target Expected output
     * @return Mean squared error value
     */
    double mse(const Matrix& predicted, const Matrix& target) {
        Matrix diff = predicted - target;
        diff.square();
        return diff.sum() / (diff.getRows() * diff.getCols());
    }

    /**
     * @brief Print network architecture
     */
    void print_architecture() const {
        std::cout << "Neural Network Architecture:" << std::endl;
        for (size_t i = 0; i < layers.size(); i++) {
            std::cout << "Layer " << i << ": " << layers[i] << " neurons" << std::endl;
        }
    }

    // Getters
    size_t getLayerCount() const { return layers.size(); }
    size_t getLayerSize(size_t layer) const { return layers[layer]; }
    double getLearningRate() const { return learning_rate; }
    void setLearningRate(double lr) { learning_rate = lr; }
};

#endif // NEURAL_NETWORK