import pandas as pd
import numpy as np
from data_spliter import data_spliter
from polynomial_model import add_polynomial_features
from ridge import MyRidge
import matplotlib.pyplot as plt


def evaluate_model(model, X, y):
    # Make predictions
    y_pred = model.predict_(X)

    # Calculate the mean squared error
    mse = np.mean((y - y_pred) ** 2)
    return mse


def benchmark_train(x, y, degrees, alphas):
    # Split the dataset into training, cross-validation, and test sets
    X_train, X_cv, X_test, y_train, y_cv, y_test = data_spliter(
        x.to_numpy(), y.to_numpy(), 0.6, 0.2)

    # Apply polynomial feature transformation to training set
    X_train_poly = add_polynomial_features(X_train, degrees[-1])

    # Apply polynomial feature transformation to cross-validation set
    X_cv_poly = add_polynomial_features(X_cv, degrees[-1])

    # Create an empty list to store the mean squared errors
    mse_values = []

    for degree in degrees:
        mse_values_alpha = []
        for alpha in alphas:
            # Create an instance of MyRidge
            num_features = degree + 1
            thetas = np.random.rand(X_train.shape[1] + 1, 1)
            ridge = MyRidge(thetas, alpha=alpha, max_iter=1000)

            # Fit the model to the training data
            ridge.fit_(X_train_poly[:, :degree+1], y_train)

            # Evaluate the model on the cross-validation data
            mse = evaluate_model(ridge, X_cv_poly[:, :degree+1], y_cv)
            mse_values_alpha.append(mse)

        mse_values.append(mse_values_alpha)

    # Find the best degree and alpha with the minimum mean squared error
    mse_values = np.array(mse_values)
    best_degree_idx, best_alpha_idx = np.unravel_index(
        np.argmin(mse_values), mse_values.shape)
    best_degree = degrees[best_degree_idx]
    best_alpha = alphas[best_alpha_idx]
    print("Best Degree:", best_degree)
    print("Best Alpha:", best_alpha)

    # Train the best model based on the best degree and alpha using the combined training and cross-validation sets
    X_train_cv_poly = add_polynomial_features(
        np.vstack((X_train, X_cv)), best_degree)
    X_test_poly = add_polynomial_features(X_test, best_degree)
    num_features = best_degree + 1
    thetas = np.zeros((num_features, 1))
    best_model = MyRidge(thetas, alpha=best_alpha, max_iter=1000)
    best_model.fit_(
        X_train_cv_poly[:, :best_degree+1], np.vstack((y_train, y_cv)))

    # Save the parameters of all the models into a file (models.csv)
    models = {'degrees': degrees, 'alphas': alphas, 'mse_values': mse_values}
    models_df = pd.DataFrame(models)
    models_df.to_csv('models.csv', index=False)
    print("Data saved to models.csv file.")

    return mse_values, best_model, X_test_poly, y_test


def plot_evaluation_curve(degrees, alphas, mse_values):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(degrees)):
        for j in range(len(alphas)):
            ax.scatter(degrees[i], alphas[j], mse_values[i, j], c='b')
    ax.set_xlabel('Degree of Polynomial')
    ax.set_ylabel('Alpha')
    ax.set_zlabel('Mean Squared Error')
    ax.set_title('Evaluation Curve')
    plt.show()


def plot_predictions(model, X, y):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], y[:, 0], c='b', label='True Price')

    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for alpha in alphas:
        model.alpha = alpha
        y_pred = model.predict_(X[:, :model.thetas.shape[0]])
        ax.scatter(X[:, 0], X[:, 1], y_pred[:, 0],
                   label='Alpha={}'.format(alpha))

    ax.set_xlabel('Weight')
    ax.set_ylabel('Production Distance')
    ax.set_zlabel('Price')
    ax.legend()
    plt.title('True Price vs Predicted Price')
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv("space_avocado.csv")

    # Extract features and target
    X = df[['weight', 'prod_distance']]
    y = df[['target']]

    # Define the degrees of polynomial to consider
    degrees = [1, 2, 3, 4]

    # Define the regularization factors to consider
    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    mse_values, best_model, X_test_poly, y_test = benchmark_train(
        X, y, degrees, alphas)

    # Plot the evaluation curve
    plot_evaluation_curve(degrees, alphas, mse_values)

    # Plot the true price and predicted price using the best model
    plot_predictions(best_model, X_test_poly, y_test)
