#include <vector>
#include <stdlib.h>
#include "neural-network.hpp"

int main(int argc, char* argv[]) {

    // Initialize neural network with 3 layers: input (2 neurons), hidden (3 neurons), output (1 neuron)
    NeuralNetwork nn({2, 3, 1}, 0.01);
    nn.print_architecture();

    return 0;
}