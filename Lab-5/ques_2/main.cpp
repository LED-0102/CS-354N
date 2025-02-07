#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>

class Perceptron {
private:
    std::vector<double> weights;
    double bias;
    double learning_rate;
    int max_epochs;

public:
    Perceptron(double lr, int epochs) : learning_rate(lr), max_epochs(epochs) {
        srand(time(0));
        weights.resize(2);
        for (double &w : weights) {
            w = ((double)rand() / RAND_MAX) * 2 - 1; // Random values between -1 and 1
        }
        bias = ((double)rand() / RAND_MAX) * 2 - 1; // Random bias
    }

    int computeOutput(const std::vector<double>& inputs) {
        double net_input = bias;
        for (size_t i = 0; i < inputs.size(); ++i) {
            net_input += inputs[i] * weights[i];
        }
        return (net_input >= 0) ? 1 : 0; // Threshold function
    }

    int train(const std::vector<std::vector<double>>& inputs, const std::vector<int>& targets) {
        int epochs = 0;
        while (epochs < max_epochs) {
            bool all_correct = true;
            for (size_t i = 0; i < inputs.size(); ++i) {
                int output = computeOutput(inputs[i]);
                if (output != targets[i]) {
                    all_correct = false;
                    for (size_t j = 0; j < weights.size(); ++j) {
                        weights[j] += learning_rate * (targets[i] - output) * inputs[i][j];
                    }
                    bias += learning_rate * (targets[i] - output);
                }
            }
            epochs++;
            if (all_correct) break;
        }
        return epochs;
    }

    void testNetwork(const std::vector<std::vector<double>>& inputs, const std::vector<int>& targets, std::ostream& out) {
        out << "\nTesting Results:\n";
        for (size_t i = 0; i < inputs.size(); ++i) {
            int output = computeOutput(inputs[i]);
            out << "Input: (" << inputs[i][0] << ", " << inputs[i][1] << ") -> Predicted: " << output << " | Target: " << targets[i];
            if (output == targets[i]) {
                out << " ✅";
            } else {
                out << " ❌";
            }
            out << std::endl;
        }
    }

    void printFinalWeights(std::ostream& out) {
        out << "\nFinal Weights: " << weights[0] << ", " << weights[1];
        out << "\nFinal Bias: " << bias << std::endl;
    }

    void interactiveTesting() {
        std::cout << "\n--- Interactive Testing Mode ---\n";
        std::cout << "Enter weight and ear length to classify as Rabbit (0) or Bear (1).\n";
        std::cout << "Type 'exit' to stop.\n";

        while (true) {
            double weight, ear_length;
            std::cout << "\nEnter weight and ear length (e.g., 3 5): ";
            if (!(std::cin >> weight >> ear_length)) {
                std::cin.clear(); // Clear input buffer
                std::string exit_command;
                std::cin >> exit_command;
                if (exit_command == "exit") {
                    std::cout << "Exiting interactive mode.\n";
                    break;
                } else {
                    std::cout << "Invalid input. Please enter two numbers or 'exit'.\n";
                    continue;
                }
            }
            std::vector<double> test_input = {weight, ear_length};
            int result = computeOutput(test_input);
            std::cout << "Predicted: " << (result == 0 ? "Rabbit 🐰" : "Bear 🐻") << std::endl;
        }
    }
};

int main() {
    std::ifstream inFile("rabbit_bear_input.txt");
    if (!inFile) {
        std::cerr << "Error: Could not open input file!" << std::endl;
        return 1;
    }

    int num_samples;
    double learning_rate;
    int max_epochs;
    inFile >> num_samples >> learning_rate >> max_epochs;

    std::vector<std::vector<double>> inputs(num_samples, std::vector<double>(2));
    std::vector<int> targets(num_samples);

    for (int i = 0; i < num_samples; ++i) {
        inFile >> inputs[i][0] >> inputs[i][1] >> targets[i];
    }
    inFile.close();

    Perceptron classifier(learning_rate, max_epochs);
    int epochs_taken = classifier.train(inputs, targets);

    std::ofstream outFile("rabbit_bear_results.txt");
    outFile << "Training completed in " << epochs_taken << " epochs.\n";
    classifier.printFinalWeights(outFile);
    classifier.testNetwork(inputs, targets, outFile);

    std::cout << "Training completed in " << epochs_taken << " epochs.\n";
    classifier.printFinalWeights(std::cout);
    classifier.testNetwork(inputs, targets, std::cout);

    // Start interactive mode
    classifier.interactiveTesting();

    return 0;
}
