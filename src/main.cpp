#include <stdlib.h>
#include "neural-network.hpp"
#include "matrix.hpp"
#include "csv_loader.hpp"

using namespace std;

int main(int argc, char* argv[]) {
    CSVLoader train_loader("../datasets/fashion-mnist_train.csv");
    CSVLoader test_loader("../datasets/fashion-mnist_test.csv");

    if (!train_loader.load() || !test_loader.load()) {
        std::cerr << "Failed to load CSV files." << std::endl;
        return 1;
    }

    // Create a neural network
    vector<size_t> layer_sizes = {784, 16, 10, 10}; // Input size, hidden layer size, output size
    double learning_rate = 0.01;
    NeuralNetwork nn(layer_sizes, learning_rate); // Input size, hidden layer size, output size
    nn.print_architecture();

    // Define training inputs and targets
    vector<Record> train_raw_data = train_loader.getData();
    vector<Record> test_raw_data = test_loader.getData();

    vector<Matrix> training_inputs;
    vector<Matrix> training_targets;

    // Convert raw data to training inputs and targets
    for (const auto& record : train_raw_data) {
        Matrix input(784, 1); // 28x28 image flattened to 784
        for (size_t i = 0; i < 784; i++) {
            input(i, 0) = record.pixels[i] / 255.0; // Normalize pixel values
        }
        training_inputs.push_back(input);

        Matrix target(10, 1); // One-hot encoding for output
        target(record.label, 0) = 1.0;
        training_targets.push_back(target);
    }

    // Train the neural network

    size_t epochs = 5;
    cout << "Training start..." << endl;
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < training_inputs.size(); ++i) {
            nn.train(training_inputs[i], training_targets[i]);
            double progress = static_cast<double>(i) / training_inputs.size() * 100.0;
            cout << "\rProgress: " << round(progress) << " %" << flush;
        }
        cout << endl;
        cout << "Epoch " << epoch + 1 << " completed." << endl;
    }

    cout << "Training completed..." << endl;

    // Test the neural network

    cout << "Testing start..." << endl;
    size_t correct_predictions = 0;
    for (size_t i = 0; i < test_raw_data.size(); ++i) {
        Matrix input(784, 1);
        for (size_t j = 0; j < 784; j++) {
            input(j, 0) = test_raw_data[i].pixels[j] / 255.0; // Normalize pixel values
        }

        Matrix output = nn.forward(input);
        size_t predicted_label = std::distance(output.getData().begin(), std::max_element(output.getData().begin(), output.getData().end()));
        
        if (predicted_label == test_raw_data[i].label) {
            correct_predictions++;
        }
    }

    double accuracy = static_cast<double>(correct_predictions) / test_raw_data.size() * 100.0;
    cout << "Testing completed..." << endl;
    cout << "Accuracy: " << accuracy << "%" << endl;


    return 0;
}