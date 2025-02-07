#include <iostream>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <sstream>

class Perceptron {
private:
    std::vector<double> weights;
    double bias;
    double learning_rate;
    int max_epochs;

public:
    Perceptron(int num_inputs, double lr, int epochs) : learning_rate(lr), max_epochs(epochs) {
        srand(time(0));
        weights.resize(num_inputs);
        for (double &w : weights) {
            w = ((double)rand() / RAND_MAX) * 2 - 1; // Random values between -1 and 1
        }
        bias = ((double)rand() / RAND_MAX) * 2 - 1; // Random bias
    }

    int computeOutput(const std::vector<int>& inputs) {
        double net_input = bias;
        for (size_t i = 0; i < inputs.size(); ++i) {
            net_input += inputs[i] * weights[i];
        }
        return (net_input >= 0) ? 1 : 0; // Threshold function f(net)
    }

    int train(const std::vector<std::vector<int>>& inputs, const std::vector<int>& targets) {
        int epochs = 0;
        bool trained=false;
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
            if (all_correct) {
                trained=true;
                break;
            }
        }
        if (!trained) return -1;
        return epochs;
    }

    void testGate(const std::string& gateName, const std::vector<std::vector<int>>& inputs, std::ostream& out) {
        out << "\n" << gateName << " Results:" << std::endl;
        for (const auto& input : inputs) {
            int output = computeOutput(input);
            out << "Input: (";
            for (size_t i = 0; i < input.size(); ++i) {
                out << input[i] << (i < input.size() - 1 ? ", " : "");
            }
            out << ") -> Output: " << output << std::endl;
        }
    }

    void printFinalWeights(std::ostream& out) {
        out << "Final Weights: ";
        for (double w : weights) out << w << " ";
        out << "\nFinal Bias: " << bias << std::endl;
    }
};

std::string generateFileName(double learning_rate) {
    std::ostringstream ss;
    ss << "output_results_lr_" << std::fixed << std::setprecision(3) << learning_rate << ".txt";
    return ss.str();
}

int main() {
    std::ifstream inFile("input.txt");
    if (!inFile) {
        std::cerr << "Error: Could not open input file!" << std::endl;
        return 1;
    }

    int num_inputs, num_samples;
    double learning_rate;
    int max_epochs;
    inFile >> num_inputs >> num_samples >> learning_rate >> max_epochs;

    std::vector<std::vector<int>> inputs(num_samples, std::vector<int>(num_inputs));
    std::vector<std::vector<int>> targets(4, std::vector<int>(num_samples));

    for (int i = 0; i < num_samples; ++i) {
        for (int j = 0; j < num_inputs; ++j) {
            inFile >> inputs[i][j];
        }
    }
    for (int g = 0; g < 4; ++g) {
        for (int i = 0; i < num_samples; ++i) {
            inFile >> targets[g][i];
        }
    }
    inFile.close();

    std::string outputFileName = generateFileName(learning_rate);
    std::ofstream outFile(outputFileName);
    
    std::vector<std::string> gateNames = {"AND", "OR", "NAND", "NOR"};
    for (size_t g = 0; g < gateNames.size(); ++g) {
        Perceptron neuron(num_inputs, learning_rate, max_epochs);
        int epochs_taken = neuron.train(inputs, targets[g]);
        outFile << "-----------------------------------" << std::endl;

        if (epochs_taken == -1) {
            outFile << gateNames[g] << " Gate could not be trained in " << max_epochs << " epochs." << std::endl;
            continue;
        }
        outFile << gateNames[g] << " Gate trained in " << epochs_taken << " epochs." << std::endl;
        neuron.printFinalWeights(outFile);
        neuron.testGate(gateNames[g] + " Gate", inputs, outFile);
    }

    std::cout << "Results saved in " << outputFileName << std::endl;
    return 0;
}
