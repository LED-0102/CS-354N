#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cassert>
#include "NeuralNetwork.h"

// Function to read CSV file (Handles both training and testing data)
void readCSVUnified(const std::string& filename, std::vector<std::vector<double>>& features, std::vector<std::vector<double>>& labels, bool has_labels) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string line;
    bool header = true;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> row;
        std::string value;

        // Skip the header row
        if (header) {
            header = false;
            continue;
        }

        // Read values
        while (std::getline(ss, value, ',')) {
            // Remove double quotes
            for (auto &ch: value) {
                if (ch == '"') {
                    ch = ' ';
                }
            }

            // Trim whitespace
            value.erase(0, value.find_first_not_of(" \t\r\n"));
            value.erase(value.find_last_not_of(" \t\r\n") + 1);

            // Check for empty values (possible trailing commas)
            if (value.empty()) {
                std::cerr << "Warning: Empty value found in " << filename << ". Skipping row." << std::endl;
                row.clear(); // Ignore the entire row
                break;
            }

            if (value == "setosa") {
                row.push_back(0.0);
            } else if (value == "versicolor") {
                row.push_back(1.0);
            } else {
                try {
                    row.push_back(std::stod(value));
                } catch (const std::invalid_argument& e) {
                    std::cerr << "Error: Non-numeric value '" << value << "' found in " << filename << std::endl;
                    exit(EXIT_FAILURE);
                }
            }
        }

        if (row.empty()) continue; // Skip empty or invalid rows

        if (has_labels) {
            labels.push_back({ row.back() }); // Extract label
            row.pop_back(); // Remove label from feature set
        }

        features.push_back(row);
    }
    file.close();
}

// Function to calculate accuracy
double calculateAccuracy(const std::vector<std::vector<double>>& features,
                         const std::vector<std::vector<double>>& labels,
                         NeuralNetwork& nn, std::ofstream& outFile) {
    int correct_predictions = 0;
    int total_predictions = static_cast<int>(features.size());

    for (size_t i = 0; i < features.size(); ++i) {
        std::vector<double> output = nn.forward(features[i]);
        int predicted = (output[0] >= 0.5) ? 1 : 0; // Convert to binary classification
        int actual = static_cast<int>(labels[i][0]);

        if (predicted == actual) {
            correct_predictions++;
            outFile << "Correct Prediction: " << predicted << "\n";
        }
        else {
            outFile << "Value differs at index " << i << ". Expected " << actual << " Predicted " << predicted << "\n";
        }
    }

    return (static_cast<double>(correct_predictions) / total_predictions) * 100.0;
}

int main(int argc, char* argv[]) {
    // Check if filenames are provided
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <training_csv> <testing_csv>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string training_filename = argv[1];
    std::string testing_filename = argv[2];

    // Dataset storage
    std::vector<std::vector<double>> features_train, labels_train;
    std::vector<std::vector<double>> features_test, labels_test;

    // Open file to write output
    std::ofstream outFile("output.txt");

    // Redirect std::cout to file
    std::streambuf* original_cout = std::cout.rdbuf();
    std::cout.rdbuf(outFile.rdbuf());

    // Read datasets
    readCSVUnified(training_filename, features_train, labels_train, true);
    readCSVUnified(testing_filename, features_test, labels_test, true);  // Test set also contains labels

    // print features train and labels train

    // for (auto &feature : features_train) {
    //     for (auto &f : feature) {
    //         std::cout << f << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // for (auto &label : labels_train) {
    //     for (auto &l : label) {
    //         std::cout << l << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // Neural Network parameters
    double learning_rate = 0.0001;
    int epochs = 35;
    int input_size = static_cast<int>(features_train[0].size());
    std::vector<int> layer_sizes = { input_size, 15, 1 }; // Single hidden layer

    // Create and train the model
    NeuralNetwork nn(layer_sizes, learning_rate);
    nn.train(features_train, labels_train, epochs);

    // Calculate accuracy on test dataset
    double accuracy = calculateAccuracy(features_test, labels_test, nn, outFile);

    std::cout << "Model Accuracy: " << accuracy << "%" << std::endl;

    // Restore std::cout to original
    std::cout.rdbuf(original_cout);

    // Close the output file
    outFile.close();

    return 0;
}
