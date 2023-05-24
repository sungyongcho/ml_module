from polynomial_model import add_polynomial_features
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mylinearregression import MyLinearRegression as MyLR

# Load the dataset
data = pd.read_csv("are_blue_pills_magics.csv")


X = np.array(data[['Micrograms']])
Y = np.array(data[['Score']])

# Define thetas for each model
theta1 = np.random.rand(2, 1)
theta2 = np.random.rand(3, 1)
theta3 = np.random.rand(4, 1)
theta4 = np.array([[-20], [160], [-80], [10], [-1]]).reshape(-1, 1)
theta5 = np.array([[1140], [-1850], [1110], [-305], [40], [-2]]).reshape(-1, 1)
theta6 = np.array([[9110], [-18015], [13400], [-4935],
                  [966], [-96.4], [3.86]]).reshape(-1, 1)

# Create instances of the models
model1 = MyLR(theta1, alpha=1e-5, max_iter=1000000)
model2 = MyLR(theta2, alpha=3e-5, max_iter=1000000)
model3 = MyLR(theta3, alpha=1e-5, max_iter=1000000)
model4 = MyLR(theta4, alpha=1e-6, max_iter=1000000)
model5 = MyLR(theta5, alpha=2.5e-8, max_iter=1000000)
model6 = MyLR(theta6, alpha=1e-9, max_iter=1000000)

# Create polynomial features for each degree
X1 = add_polynomial_features(X, 1)
X2 = add_polynomial_features(X, 2)
X3 = add_polynomial_features(X, 3)
X4 = add_polynomial_features(X, 4)
X5 = add_polynomial_features(X, 5)
X6 = add_polynomial_features(X, 6)


# Fit the models
model1.fit_(X1, Y)
model2.fit_(X2, Y)
model3.fit_(X3, Y)
model4.fit_(X4, Y)
model5.fit_(X5, Y)
model6.fit_(X6, Y)

# Calculate MSE scores
mse_scores = [model1.mse_(X1, Y),
              model2.mse_(X2, Y),
              model3.mse_(X3, Y),
              model4.mse_(X4, Y),
              model5.mse_(X5, Y),
              model6.mse_(X6, Y)]

for i, mse_score in enumerate(mse_scores):
    print(f"mse_score for model{i+1}: {mse_score}")

# Plot bar plot for MSE scores
degrees = [1, 2, 3, 4, 5, 6]
plt.bar(degrees, mse_scores)
plt.xlabel('Polynomial Degree')
plt.ylabel('MSE Score')
plt.title('MSE Score of Polynomial Regression Models')
plt.show()

# Plot data points and models
prediction_points = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
plt.scatter(X, Y, label='Data Points')

plt.plot(prediction_points, model1.predict_(
    add_polynomial_features(prediction_points, 1)), label='Degree 1')
plt.plot(prediction_points, model2.predict_(
    add_polynomial_features(prediction_points, 2)), label='Degree 2')
plt.plot(prediction_points, model3.predict_(
    add_polynomial_features(prediction_points, 3)), label='Degree 3')
plt.plot(prediction_points, model4.predict_(
    add_polynomial_features(prediction_points, 4)), label='Degree 4')
plt.plot(prediction_points, model5.predict_(
    add_polynomial_features(prediction_points, 5)), label='Degree 5')
plt.plot(prediction_points, model6.predict_(
    add_polynomial_features(prediction_points, 6)), label='Degree 6')

plt.xlabel('Micrograms')
plt.ylabel('Score')
plt.title('Polynomial Regression Models')
plt.legend()
plt.grid(True)
plt.show()
