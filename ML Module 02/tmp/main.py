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

def plot_evaluation_curve(degrees, mse_values):
    plt.plot(degrees, mse_values, 'bo-')
    plt.xlabel('Degree of Polynomial')
    plt.ylabel('Mean Squared Error')
    plt.title('Evaluation Curve')
    plt.show()

def plot_predictions(model, X_test_poly, y_test):
    y_pred = model.predict_(X_test_poly)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_test_poly[:, 1], X_test_poly[:, 2], y_test[:, 0], c='b', label='True Price')
    ax.scatter(X_test_poly[:, 1], X_test_poly[:, 2], y_pred[:, 0], c='r', label='Predicted Price')
    ax.set_xlabel('Weight')
    ax.set_ylabel('Production Distance')
    ax.set_zlabel('Price')
    ax.legend()
    plt.title('True Price vs Predicted Price')
    plt.show()

def benchmark_train():
    # Load the dataset
    df = pd.read_csv("space_avocado.csv")

    # Split the dataset into training and test sets
    X = df[['weight', 'prod_distance', 'time_delivery']]
    y = df[['target']]
    X_train, X_test, y_train, y_test = data_spliter(X.to_numpy(), y.to_numpy(), 0.8)

    # Apply feature scaling to training and test sets
    X_train_scaled = (X_train - X_train.min()) / (X_train.max() - X_train.min())
    X_test_scaled = (X_test - X_train.min()) / (X_train.max() - X_train.min())

    # Create an empty list to store the mean squared errors
    mse_values = []

    # Define the degrees of polynomial to consider
    degrees = [1, 2, 3, 4]

    for degree in degrees:
        # Apply polynomial feature transformation to training set
        X_train_poly = add_polynomial_features(X_train_scaled, degree)

        # Apply polynomial features transformation to test set as well
        X_test_poly = add_polynomial_features(X_test_scaled, degree)

        # Create an instance of MyLinearRegression
        num_features = X_train_poly.shape[1] + 1  # Add 1 for the intercept term
        thetas = np.zeros((num_features, 1))
        lr = MyLinearRegression(thetas, alpha=0.01, max_iter=1000)

        # Fit the model to the training data
        lr.fit_(X_train_poly, y_train)

        # Evaluate the model on the test data
        mse = evaluate_model(lr, X_test_poly, y_test)
        mse_values.append(mse)

    # Find the best degree with the minimum mean squared error
    best_degree = degrees[np.argmin(mse_values)]
    print("Best Degree:", best_degree)

    # Plot the evaluation curve
    plot_evaluation_curve(degrees, mse_values)

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

    # Plot the true price and predicted price using the best model
    plot_predictions(best_model, X_test_poly, y_test)

benchmark_train()
