import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Loading the data
iris = load_iris()
X = iris.data[:,:2]   # All rows and first 2 columns
y = (iris.target == 0).astype(int)  # compare and get Y for binary classification   # .astype() changes to int

class Perceptron:

  # Initializing function that gets executed on the starting itself

  def __init__(self, lr = 0.01, epochs = 50):
      print("Initializing the Perceptron...")
      self.lr = lr  # Learning Rate
      self.epochs = epochs  # Number of training iterations
      self.weights = None   # Model Weights
      self.bias = None      # Bias term

  def fit(self, X, y):
      n_samples, n_features = X.shape

    # Initializing weights and bias...
      self.weights = np.zeros(n_features)
      self.bias = 0


      for epoch in range(self.epochs):
        for idx, x_i in enumerate(X):   # idx is serial number and x_i is the input (The two features in this case)
          # Compute Linear output of Prediction
          linear_output = np.dot(x_i, self.weights) + self.bias
          y_pred = 1 if linear_output >= 0 else 0


          # Update weights and bias
          update = self.lr * (y[idx] - y_pred)
          self.weights += update*x_i
          self.bias += update

        #self.plot_decision_boundary(X, y, epoch)
        #print(f"y[idx]:{y[idx]}\ny_pred:{y_pred}\nupdate:{update}\nWeight:{self.weights}\nbias:{self.bias}")
        # plot decision boundary to visualize training progress

        if epoch % 10 == 0:   # Mistake 2: Made the indentation error aggaaaiiiinnnnn!!!! Got the loop executed again and again in the same form...
          self.plot_decision_boundary(X, y, epoch)

  # makes predictions by computing the linear combination of input features
  def predict(self, X):
      linear_output = np.dot(X, self.weights) + self.bias

    # checking whether the output is above or below zero...
      return np.where(linear_output >= 0, 1, 0)


  def plot_decision_boundary(self, X, y, epoch):
      if self.weights[1] == 0:
        return  # avoid division by zero
      plt.figure()    # Mistake 1 : passed all the important statements under the return statement, causing the return statement to execute and close the function...

      # Scatter plot for points
      plt.scatter(X[:,0], X[:,1], c = y, cmap = 'bwr', alpha = 0.7)

      # Line Plot for the decision boundary
      x_vals = np.linspace(X[:,0].min(), X[:,0].max(), 100)
      y_vals = -(self.weights[0] * x_vals + self.bias) / self.weights[1]
      plt.plot(x_vals, y_vals, 'k')

      # Generic
      plt.title(f'Perceptron Algorithm - Epoch {epoch + 1}')
      plt.xlabel('Sepal Length')
      plt.ylabel('Sepal Width')
      plt.show()

ptron = Perceptron(0.01,200)
ptron.fit(X,y)