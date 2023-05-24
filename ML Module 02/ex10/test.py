import pandas as pd
import numpy as np
from data_spliter import data_spliter
from polynomial_model import add_polynomial_features
from mylinearregression import MyLinearRegression
import matplotlib.pyplot as plt

from benchmark_train import benchmark_train

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


if __name__ == "__main__":
    df = pd.read_csv("space_avocado.csv")

    # Split the dataset into training and test sets
    X = df[['weight', 'prod_distance', 'time_delivery']]
    y = df[['target']]
    # Define the degrees of polynomial to consider
    degrees = [1, 2, 3, 4]

    mse_values, best_model, X_test_poly, y_test =\
        benchmark_train(X, y, degrees)

    # Plot the evaluation curve
    plot_evaluation_curve(degrees, mse_values)


    # load the saved

    # Plot the true price and predicted price using the best model
    plot_predictions(best_model, X_test_poly, y_test)



