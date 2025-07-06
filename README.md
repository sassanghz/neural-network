# MNIST Digit Classifier (Neural Network from Scratch)

This project implements a simple **feedforward neural network** using **NumPy** to classify handwritten digits (0–9) from the **MNIST dataset**.

The network is trained from scratch — no machine learning libraries like TensorFlow or PyTorch are used. It's a great educational example of how neural networks work internally.

---

## 📦 Features

- Uses the MNIST dataset via `fetch_openml`
- Input layer: 784 nodes (28×28 pixels)
- Hidden layer: 64 neurons
- Output layer: 10 neurons (digits 0–9)
- Activation: Sigmoid function
- Loss: Mean Squared Error (MSE)
- Training: Manual backpropagation with gradient descent
- Visualizes accuracy and loss over epochs
- Can display prediction on a test image

---

## 🛠 Requirements

- Python 3.x
- NumPy
- Matplotlib
- Scikit-learn
- Pandas

You can install the requirements using:

```bash
pip install numpy matplotlib scikit-learn pandas

