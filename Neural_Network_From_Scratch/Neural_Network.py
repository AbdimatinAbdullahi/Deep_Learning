import numpy as np


# Relu activation function
def relu(z):
    return np.maximum(0, z)


# Sigmoid Activation function
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


# Derivative of sigmoid
def sigmoid_derivative(z):
    sigmoid(z) * (1 - sigmoid(z))


# Derivative of sigmoid
def relu_derivatives(z):
    return np.where(z > 0, 1, 0)


def neural_network(X, Y, hidden_size=2, learning_rate=0.1, iterations=100000):
    # Getting the Input Size and output size
    input_size = X.shape[0]
    output_size = Y.shape[0]

    # Initialize weights and bias
    np.random.seed(1)
    w1 = np.random.randn(hidden_size, input_size) * 0.01  # Weights for input layer
    b1 = np.random.randn(hidden_size, 1)
    w2 = np.random.randn(output_size, hidden_size) * 0.01  # Weights for output layer
    b2 = np.random.randn(output_size, 1)

    for i in range(iterations):

        z1 = np.dot(w1, X) + b1
        a1 = relu(z1)
        z2 = np.dot(w2, a1) + b2
        a2 = sigmoid(z2)

        # Computing the loss (Binary cross entropy)
        m = Y.shape[1]
        epsilon = 1e-15
        loss = -1 / m * (np.dot(Y, np.log(a2 + epsilon).T) + np.dot(1 - Y, np.log(1 - a2 + epsilon).T))
        loss = np.squeeze(loss)

        if i % 1000 == 0:
            print(f'Iterations: {iterations}: Loss: {loss}')

        # Backward Propagation
        dz2 = a2 - Y
        dw2 = 1 / m * np.dot(dz2, a1.T)
        db2 = 1 / m * np.sum(dz2, axis=1,keepdims=True)
        da1 = np.dot(w2.T, dz2)
        dz1 = da1 * relu_derivatives(z1)
        dw1 = 1 / m * np.dot(dz1, X.T)
        db1 = 1 / m * np.sum(dz1, axis=1 ,keepdims=True)

        # Updating parameters
        w1 -= learning_rate * dw1
        b1 -= learning_rate * db1
        w2 -= learning_rate * dw2
        b2 -= learning_rate * db2

    # Making Predictions
    z1 = np.dot(w1, X) + b1
    a1 = relu(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = sigmoid(z2)

    predictions = (a2 > 0.5).astype(int)
    return predictions, w1, b1, w2, b2


X = np.array([[20, 25, 45], [40000, 100000, 120000]]) / 1000
Y = np.array([[0, 1, 1]])

predictions, w1, b1, w2, b2 = neural_network(X, Y, hidden_size=4, iterations=100000)
print(f'Weight 1, {w1}')
print(f'Weight 2, {w2}')
print(f'Bias 1, {b1}')
print(f'Bias 2, {b2}')
print(predictions)
