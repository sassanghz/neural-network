#visualizing the dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

mnist = fetch_openml('mnist_784', version = 1) # loading mnist ds
X = mnist.data / 255.0 # normalization for input img
Y = np.array(mnist.target.astype(int)).reshape(-1,1) # target digits, reshaping vectors

encoder = OneHotEncoder(sparse_output=False)
Y_encoded = encoder.fit_transform(Y)

#train/test
x_train, x_test, y_train, y_test = train_test_split(
    np.array(X[:5000]), 
    np.array(Y_encoded[:5000]), 
    test_size=0.2
)

# the encoders would use the first 5000 samples
# only 20% will be used for testing

input_size = 784 # 28x28
hidden = 64 # neurons in hidden layer
output = 10 # between 0-9

#activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x)) # values between 0 and 1

def forward_pass(x, w1, w2):
    z1 = x.dot(w1)
    a1 = sigmoid(z1)
    z2 = a1.dot(w2)
    a2 = sigmoid(z2)

    return a2 # calculation of the weights

def generate_weight(x, y):
    return np.random.randn(x,y) * 0.1

# mse
def loss(out, y_true):
    return np.mean(np.square(out - y_true)) # loss between predicted values and true

#backprop
def backPropagation(x, y, w1, w2, alpha):
    z1 = x.dot(w1)
    a1 = sigmoid(z1)
    z2 = a1.dot(w2)
    a2 = sigmoid(z2)

    d2 = a2 - y
    d1 = np.multiply(d2.dot(w2.T), a1 * (1 - a1))

    w1 -= alpha * x.T.dot(d1)
    w2 -= alpha * a1.T.dot(d2)

    return w1, w2
    # gradient loss weights, using gradient descent with alpha 

w1 = generate_weight(input_size, hidden) # input matrice
w2 = generate_weight(hidden, output) # output matrice

def train(x, y, w1, w2, alpha=0.1, epochs=20):
    acc_list = []
    loss_list = []
    
    for epoch in range(epochs):
        losses = []
        
        for i in range(len(x)):
            xi = x[i].reshape(1, -1)
            yi = y[i].reshape(1, -1)
            out = forward_pass(xi, w1, w2)
            losses.append(loss(out, yi))
            w1, w2 = backPropagation(xi, yi, w1, w2, alpha)
        
        avg_loss = np.mean(losses)

        accuracy = evaluate(x, y, w1, w2)

        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        acc_list.append(accuracy)
        loss_list.append(avg_loss)
    
    return acc_list, loss_list, w1, w2

def evaluate(x, y, w1, w2):
    correct = 0

    for i in range(len(x)):
        out = forward_pass(x[i].reshape(1,-1), w1, w2)

        if np.argmax(out) == np.argmax(y[i]):
            correct += 1
    
    return (correct / len(x)) * 100

acc, loss, w1, w2 = train(x_train, y_train, w1, w2, alpha=0.1, epochs = 20)

# accuracy and loss display
plt.plot(acc)
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.show()

plt.plot(loss)
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

def predict_values(index):
    sample = x_test[index].reshape(1, -1) # picking an image from the test
    out = forward_pass(sample, w1, w2)
    prediction = np.argmax(out)

    plt.imshow(x_test[index].reshape(28,28), cmap='gray')
    # display prediction
    plt.title(f"Predicted Output: {prediction}")
    plt.axis('off')
    plt.savefig("output.png")

predict_values(7)