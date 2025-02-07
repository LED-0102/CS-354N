# 3-Input Perceptron for Logic Gates

This project implements a **3-input perceptron neural network** to learn and classify the following logic gates:
- **AND**
- **OR**
- **NAND**
- **NOR**

## 1. Requirements
- C++ compiler (e.g., g++)

## 2. Input Structure
The input file (`input.txt`) should contain:
1. **Number of inputs** (integer, always 3)
2. **Number of samples** (integer)
3. **Learning rate** (float)
4. **Max epochs** (integer)
5. **Training data**: Each row has:  

    input1, input2, input3


6. **Expected outputs** for all four gates:  
AND targets  
OR targets  
NAND targets  
NOR targets 


### Example (`input.txt`)
3 8 0.0001 1000  
0 0 0  
0 0 1  
0 1 0  
0 1 1  
1 0 0  
1 0 1  
1 1 0  
1 1 1  
0 0 0 0 0 0 0 1  
0 1 1 1 1 1 1 1  
1 1 1 1 1 1 1 0  
1 0 0 0 0 0 0 0  


## 3. Running the Training
1. **Compile the program**:  
```
g++ main.cpp -o main
```

2. **Run the program**:
```
./main
```

3. The output file **`output_results_<learning_rate>.txt`** will be created, containing:
- Number of epochs required for convergence
- Final weights and bias
- Predictions for each logic gate

## 4. Expected Output Example
```
AND Gate trained in 12 epochs. 
Final Weights: 0.45, 0.67, 0.23 
Final Bias: -0.15

AND Gate Results: 
Input: (0, 0, 0) -> Output: 0 
Input: (0, 0, 1) -> Output: 0 
Input: (1, 1, 1) -> Output: 1 ...
```
  

## 5. Interactive Testing
This program **does not include interactive testing** but evaluates the training data automatically.

## 6. Troubleshooting
- **"Error: Could not open input file!"** → Ensure `input.txt` exists.
- **Wrong predictions?** → Try different **learning rates**.
- **Too many epochs?** → Adjust `max_epochs` or change initialization.

---
✅ **Trains on all four logic gates**  
✅ **Saves results in an output file**  
✅ **Automatically determines convergence**  

🚀 **Run the program and test logic gates using a perceptron!** 🚀


