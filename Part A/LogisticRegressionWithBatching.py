from script1 import Linear ,sigmoid, bce
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
# Define constants hyperparamters
CLASS1_SIZE = 100
CLASS2_SIZE = 100
N_FEATURES = 2
N_OUTPUT = 1
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
# Build computation graph
epochs = 100
learning_rate = 0.01
u_node=Linear(np.transpose([W1,W2]) ,W0,learning_rate)
batch_size=1
Sigmoid=sigmoid()
BCE=bce()
# Forward and Backward Pass
def forward_pass(u_node,Sigmoid,BCE , x,y):# Manual forward pass through 3-layer graph: Linear -> Sigmoid -> BCE
    u_node.forward(x)
    ylocal=u_node.predy
    Sigmoid.forward(ylocal)
    yh=Sigmoid.sigforward
    BCE.forward(yh,y)
def backward_pass(u_node,Sigmoid,BCE):# Backward pass: BCE -> Sigmoid -> Linear (chain rule)
    sigback=BCE.backward()
    linearback=Sigmoid.backward(sigback)
    u_node.backward(linearback)
for i in range(epochs):# Training loop Handles Batching now
    total_loss=0
    for j in range(0,len(X_train),batch_size):
        x=X_train[j:j+batch_size]
        y=y_train[j:j+batch_size]
        x=np.transpose(x)
        forward_pass(u_node,Sigmoid,BCE,x,y)
        backward_pass(u_node,Sigmoid,BCE)
        total_loss+=BCE.loss
    print(f"Epoch {i+1}, Loss: {total_loss / len(X_train)}")

#Evaluation with batch
correct_predictions = 0
for i in range(0,len(X_test),batch_size):
    x=X_test[i:i+batch_size]
    y=y_test[i:i+batch_size]
    x=np.transpose(x)
    forward_pass(u_node,Sigmoid,BCE,x,y)
    for element in Sigmoid.sigforward:
        for j, number in enumerate(element):
            if number>0.5 and y[j]==1 :# Binary classification - compare each prediction with true label
                correct_predictions+=1
            elif number<=0.5 and y[j]==0:
                correct_predictions+=1
accuracy = correct_predictions / X_test.shape[0]
print(f"Accuracy: {accuracy * 100:.2f}%")

#Testing The Effect of batch size on trainig loss
batch_sizes = [1, 2, 4, 8, 16, 32, len(X_train)]
loss_history = {}
for batch_size in batch_sizes:
    print(f"\n===== Training with batch size = {batch_size} =====")
    W0 = np.zeros(1)
    W1 = np.random.randn(1) * 0.1
    W2 = np.random.randn(1) * 0.1
    u_node = Linear(np.transpose([W1, W2]), W0, learning_rate)
    Sigmoid = sigmoid()
    BCE = bce()
    losses = []
    for epoch in range(epochs):
        total_loss = 0
        for j in range(0, len(X_train), batch_size):
            x = np.transpose(X_train[j:j+batch_size])
            y_batch = y_train[j:j+batch_size]
            forward_pass(u_node, Sigmoid, BCE, x, y_batch)
            backward_pass(u_node, Sigmoid, BCE)
            total_loss += BCE.loss
        avg_loss = total_loss / len(X_train)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    loss_history[batch_size] = losses
#Plotting the various training loss
plt.figure(figsize=(8,6))
for bs, losses in loss_history.items():
    plt.plot(losses, label=f'Batch size {bs}')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Effect of Batch Size on Training Loss')
plt.legend()
plt.grid(True)
plt.show()
