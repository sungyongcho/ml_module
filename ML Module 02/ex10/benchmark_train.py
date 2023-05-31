import pandas as pd
import numpy as np
from data_spliter import data_spliter
from polynomial_model import add_polynomial_features
from mylinearregression import MyLinearRegression
import matplotlib.pyplot as plt


def evaluate_model(model, X_test_poly, y_test):
    # Make predictions on the test data
    y_pred = model.predict_(X_test_poly)

    # Calculate the mean squared error
    mse = np.mean((y_test - y_pred) ** 2)
    return mse


def benchmark_train(x, y, degrees):
    '''
    returns mse values
    '''

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = data_spliter(
        x.to_numpy(), y.to_numpy(), 0.8)

    # Apply feature scaling to training and test sets
    X_train_scaled = (X_train - X_train.min()) / \
        (X_train.max() - X_train.min())
    X_test_scaled = (X_test - X_train.min()) / (X_train.max() - X_train.min())

    # Create an empty list to store the mean squared errors
    mse_values = []

    for degree in degrees:
        # Apply polynomial feature transformation to training set
        X_train_poly = add_polynomial_features(X_train_scaled, degree)

        # Apply polynomial features transformation to test set as well
        X_test_poly = add_polynomial_features(X_test_scaled, degree)

        # Create an instance of MyLinearRegression
        # Add 1 for the intercept term
        num_features = X_train_poly.shape[1] + 1
        thetas = np.zeros((num_features, 1))
        lr = MyLinearRegression(thetas, alpha=0.01, max_iter=1000)

        # Fit the model to the training data
        lr.fit_(X_train_poly, y_train)

        # Evaluate the model on the test data
        mse = evaluate_model(lr, X_test_poly, y_test)
        mse_values.append(mse)

    # Find the best degree with the minimum mean squared error
    print(degrees, mse_values)
    best_degree = degrees[np.argmin(mse_values)]
    print("Best Degree:", best_degree)

    # Train the best model based on the best degree
    X_train_poly = add_polynomial_features(X_train_scaled, best_degree)
    X_test_poly = add_polynomial_features(X_test_scaled, best_degree)
    num_features = X_train_poly.shape[1] + 1
    thetas = np.zeros((num_features, 1))
    best_model = MyLinearRegression(thetas, alpha=0.01, max_iter=1000)
    best_model.fit_(X_train_poly, y_train)

    # Save the parameters of all the models into a file (models.csv)
    models = {'degrees': degrees, 'mse_values': mse_values}
    models_df = pd.DataFrame(models)
    models_df.to_csv('models.csv', index=False)
    print("!!data saved to models.csv file in same directory.!!")

    return mse_values, best_model, X_test_poly, y_test
