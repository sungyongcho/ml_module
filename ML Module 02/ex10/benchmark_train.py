from mylinearregression import MyLinearRegression as MyLR
from data_spliter import data_spliter
from polynomial_model import add_polynomial_features
import pandas as pd
import numpy as np


# Load the dataset
df = pd.read_csv("space_avocado.csv")

# Split the dataset into training and test sets
X = df[['weight', 'prod_distance', 'time_delivery']]
y = df[['target']]
X_train, X_test, y_train, y_test = data_spliter(
    X.to_numpy(), y.to_numpy(), 1.0)

print(y_train.shape)

# Apply polynomial feature transformation to training set
X_train_poly = add_polynomial_features(X_train, 4)
print(len(X))
print(X_train_poly.shape)


degrees = range(1, 5)  # Consider degrees from 1 to 4
mse_scores = []


# Apply polynomial features transformation to test set as well
X_test_poly = add_polynomial_features(X_test, 4)
print(X_test_poly.shape)

# Create an instance of MyLinearRegression
thetas = np.zeros((X_train_poly.shape[1] - 1, 1))
print(thetas.shape)
lr = MyLR(thetas, alpha=0.001, max_iter=1000)
print(lr.thetas)

# Fit the model to the training data
print(len(X_train_poly), len(y_train))
lr.fit_(X_train_poly, y_train)

# # # # Make predictions on the test data
# # y_pred = lr.predict_(X_test_poly)

# # print(np.array(y_pred))

# # # Calculate the mean squared error
# # mse = lr.mse_(y_test, np.array(y_pred))

# # mse_scores.append(mse)


# # best_degree = degrees[np.argmin(mse_scores)]
# # print("Best hypothesis degree:", best_degree)
