from LinearClass import Linear ,sigmoid, bce
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
# Define constants hyperparamters
CLASS1_SIZE = 100
CLASS2_SIZE = 100
N_FEATURES = 2
N_OUTPUT = 1
LEARNING_RATE = 0.02
EPOCHS = 100
TEST_SIZE = 0.25

# Define the means and covariances of the two components
MEAN1 = np.array([1, 2])
COV1 = np.array([[1, 0], [0, 1]])
MEAN2 = np.array([1, -2])
COV2 = np.array([[1, 0], [0, 1]])

# Generate random points from the two components
X1 = multivariate_normal.rvs(MEAN1, COV1, CLASS1_SIZE)
X2 = multivariate_normal.rvs(MEAN2, COV2, CLASS2_SIZE)

# Combine the points and generate labels
X = np.vstack((X1, X2))
y = np.hstack((np.zeros(CLASS1_SIZE), np.ones(CLASS2_SIZE)))

# Plot the generated data
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot of Generated Data')
plt.show()

# Split data
indices = np.arange(X.shape[0])
np.random.shuffle(indices)

test_set_size = int(len(X) * TEST_SIZE)
test_indices = indices[:test_set_size]
train_indices = indices[test_set_size:]

X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

# Model parameters
n_features = X_train.shape[1]
n_output = 1

# Initialize weights and biases
W0 = np.zeros(1)
W1 = np.random.randn(1) * 0.1
W2 = np.random.randn(1) * 0.1
#b1_node = Parameter(b1)

# Build computation graph
epochs = 100
learning_rate = 0.01
u_node=Linear(np.transpose([W1,W2]) ,W0,learning_rate)
Sigmoid=sigmoid()
BCE=bce()
# Forward and Backward Pass
def forward_pass(u_node,Sigmoid,BCE , x,y):
    u_node.forward(x)
    yhloc=u_node.predy
    Sigmoid.forward(yhloc)
    yh=Sigmoid.sigforward
    BCE.forward(yh,y)
def backward_pass(u_node,Sigmoid,BCE):
    sigback=BCE.backward()
    linearback=Sigmoid.backward(sigback)
    u_node.backward(linearback)
for i in range(epochs):
    total_loss=0
    for j in range(len(X_train)):
        x=X_train[j]
        y=y_train[j]
        forward_pass(u_node,Sigmoid,BCE,x,y)
        backward_pass(u_node,Sigmoid,BCE)
        total_loss+=BCE.loss
    print(f"Epoch {i+1}, Loss: {total_loss / len(X_train)}")

# Evaluate the model
correct_predictions = 0
for i in range(len(X_test)):
    x=X_test[i]
    y=y_test[i]
    forward_pass(u_node,Sigmoid,BCE,x,y)
    if Sigmoid.sigforward>0.5:
        correct_predictions+=1
accuracy = correct_predictions / X_test.shape[0]
print(f"Accuracy: {accuracy * 100:.2f}%")
