'''
Author, date:
xinghua liu, 20230117

Description:
reference
Mini-Batch Gradient Descent (MBGD):
https://www.geeksforgeeks.org/ml-mini-batch-gradient-descent-with-python/

A simple implementation to learn how to use MBGD in neural network.
We use this method to compute the model parameters (coefficients and bias) for algorithms like linear regression, logistic regression, neural networks, etc. network to predict the Gender (y_ture) of
a given intput data sample with Weight (x1) and Height(x1).

Description:
Let theta = model parameters and max_iters = number of epochs. for itr = 1, 2, 3, …, max_iters:       for mini_batch (X_mini, y_mini):
(1)Forward Pass on the batch X_mini:
 - Make predictions on the mini-batch;
 - Compute error in predictions (J(theta)) with the current values of the parameters
(2)Backward Pass:
 - Compute gradient(theta) = partial derivative of J(theta) w.r.t. theta
(3)Update parameters:
 - theta = theta - learning_rate*gradient(theta)

Depending on the number of training examples considered in updating the model parameters, we have the following 4 types of gradient descents:
(1)Batch Gradient Descent:
Parameters are updated after computing the gradient of the error with respect to the entire training set.
(2)Stochastic Gradient Descent:
Parameters are updated after computing the gradient of the error with respect to a single training example.
(3)Mini-Batch Gradient Descent:
Parameters are updated after computing the gradient of  the error with respect to a subset of the training set.
(4)Mini-Batch Stochastic Gradient Descent:
The most important and maybe the only way to update parameters of a very large neural network which includes the advantages of previous 3-types of gradient descent methods.
'''

# importing dependencies
import numpy as np
import matplotlib.pyplot as plt

# creating data
mean = np.array([5.0, 6.0])
cov = np.array([[1.0, 0.95], [0.95, 1.2]])
data = np.random.multivariate_normal(mean, cov, 8000)

# visualising data
plt.scatter(data[:500, 0], data[:500, 1], marker='.')
plt.show()

# train-test-split
data = np.hstack((np.ones((data.shape[0], 1)), data))

split_factor = 0.90
split = int(split_factor * data.shape[0])

X_train = data[:split, :-1]
y_train = data[:split, -1].reshape((-1, 1))
X_test = data[split:, :-1]
y_test = data[split:, -1].reshape((-1, 1))

print("Number of examples in training set= %d " % (X_train.shape[0]))
print("Number of examples in testing set= % d " % (X_test.shape[0]))


# linear regression using "mini-batch" gradient descent
# function to compute hypothesis / predictions
def hypothesis(X, theta):
  return np.dot(X, theta)

# function to compute gradient of error function w.r.t. theta
def gradient(X, y, theta):
  h = hypothesis(X, theta)
  grad = np.dot(X.transpose(), (h - y))
  return grad

# function to compute the error for current values of theta
def cost(X, y, theta):
  h = hypothesis(X, theta)
  J = np.dot((h - y).transpose(), (h - y))
  J /= 2
  return J[0]

# function to create a list containing mini-batches
def create_mini_batches(X, y, batch_size):
  mini_batches = []
  data = np.hstack((X, y))
  np.random.shuffle(data)
  n_minibatches = data.shape[0] // batch_size
  i = 0

  for i in range(n_minibatches + 1):
    mini_batch = data[i * batch_size:(i + 1)*batch_size, :]
    X_mini = mini_batch[:, :-1]
    Y_mini = mini_batch[:, -1].reshape((-1, 1))
    mini_batches.append((X_mini, Y_mini))
  if data.shape[0] % batch_size != 0:
    mini_batch = data[i * batch_size:data.shape[0]]
    X_mini = mini_batch[:, :-1]
    Y_mini = mini_batch[:, -1].reshape((-1, 1))
    mini_batches.append((X_mini, Y_mini))
  return mini_batches

# function to perform mini-batch gradient descent
def gradientDescent(X, y, learning_rate=0.001, batch_size=32):
  theta = np.zeros((X.shape[1], 1))
  error_list = []
  max_iters = 3
  for itr in range(max_iters):
    mini_batches = create_mini_batches(X, y, batch_size)
    for mini_batch in mini_batches:
      X_mini, y_mini = mini_batch
      theta = theta - learning_rate * gradient(X_mini, y_mini, theta)
      error_list.append(cost(X_mini, y_mini, theta))

  return theta, error_list

# Compute the model parameters (theta) and visualize the change in the error function
theta, error_list = gradientDescent(X_train, y_train)
print("Bias = ", theta[0])
print("Coefficients = ", theta[1:])


# visualising gradient descent
plt.plot(error_list)
plt.xlabel("Number of iterations")
plt.ylabel("Cost")
plt.show()


# Make predictions on the testing set and compute the mean absolute error in predictions
# predicting output for X_test
y_pred = hypothesis(X_test, theta)
plt.scatter(X_test[:, 1], y_test[:, ], marker='.')
plt.plot(X_test[:, 1], y_pred, color='orange')
plt.show()

# calculating error in predictions
error = np.sum(np.abs(y_test - y_pred) / y_test.shape[0])
print("Mean absolute error = " , error)
#The orange line represents the final hypothesis function: theta[0] + theta[1]*X_test[:, 1] + theta[2]*X_test[:, 2] = 0
