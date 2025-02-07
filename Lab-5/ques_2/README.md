# Toy Animal Classifier (Rabbit vs. Bear) - Perceptron Model

This project implements a **Perceptron Neural Network** to classify toy animals as either **rabbits (0)** or **bears (1)** based on their **weight** and **ear length**.

## 1. Requirements
- C++ compiler (e.g., g++)

## 2. Input Structure
The input file (`rabbit_bear_input.txt`) should contain:
1. **Number of samples** (integer)
2. **Learning rate** (float)
3. **Max epochs** (integer)
4. **Training data**: Each row has: weight, ear length, target (0 = rabbit, 1 = bear)


### Example (`rabbit_bear_input.txt`)
8 0.01 1000  
1 4 0  
1 5 0  
2 4 0  
2 5 0  
3 1 1  
3 2 1  
4 1 1  
4 2 1  


## 3. Running the Training
1. **Compile the program**:

```
g++ main.cpp -o perceptron
```

2. **Run the program**:

```
./perceptron
```

3. The output file `rabbit_bear_results.txt` will be created with:
- Number of epochs taken
- Final weights and bias
- Testing results on training data

## 4. Interactive Testing (After Training)
After training, the program will prompt for new inputs to classify:

```
Enter weight and ear length: 3 3 Predicted Class: 1 (Bear)
```


## 5. Expected Output Example

```
Training completed in 5 epochs.

Final Weights: 1.23, -2.45 Final Bias: 0.678

Testing Results: Input: (1, 4) -> Predicted: 0 | Target: 0 ✅ Input: (1, 5) -> Predicted: 0 | Target: 0 ✅ Input: (3, 1) -> Predicted: 1 | Target: 1 ✅ ...
```


