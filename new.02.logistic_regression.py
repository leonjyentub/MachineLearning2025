import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compute_cost(X, y, weights):
    m = len(y)
    h = sigmoid(np.dot(X, weights))
    cost = (-1/m) * np.sum(y * np.log(h) + (1-y) * np.log(1-h))
    return cost


def gradient_descent(X, y, weights, learning_rate, num_iterations):
    m = len(y)
    cost_history = []

    for i in range(num_iterations):
        h = sigmoid(np.dot(X, weights))
        gradient = np.dot(X.T, (h - y)) / m
        weights -= learning_rate * gradient
        cost = compute_cost(X, y, weights)
        cost_history.append(cost)

    return weights, cost_history


# Load the data
data = pd.read_csv('student_exam_data.csv')
X = data['study_hours'].values
y = data['pass_fail'].values

# Add a column of ones to X (for the bias term)
X = np.c_[np.ones(X.shape[0]), X]

# Initialize weights
weights = np.zeros(X.shape[1])

# Set hyperparameters
learning_rate = 0.01
num_iterations = 1000

# Run gradient descent
weights, cost_history = gradient_descent(
    X, y, weights, learning_rate, num_iterations)

# Plot the decision boundary
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 1], y, c=y, cmap='viridis', edgecolors='k')
x_boundary = np.array([X[:, 1].min(), X[:, 1].max()])
y_boundary = -(weights[0] + weights[1] * x_boundary) / weights[1]
plt.plot(x_boundary, sigmoid(y_boundary), 'r-', lw=2)
plt.xlabel('Study Hours')
plt.ylabel('Pass/Fail')
plt.title('Logistic Regression Decision Boundary')
plt.savefig('decision_boundary.png')
plt.close()

# Plot the cost history
plt.figure(figsize=(10, 6))
plt.plot(cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost History')
plt.savefig('cost_history.png')
plt.close()

print(f"Final weights: {weights}")
print(f"Final cost: {cost_history[-1]}")

# Make predictions


def predict(X, weights):
    return sigmoid(np.dot(X, weights)) >= 0.5


accuracy = np.mean(predict(X, weights) == y)
print(f"Accuracy: {accuracy:.2f}")
